require 'onnx/chainer/operators/gemm'
require 'onnx/chainer/operators/relu'

module Onnx
  module Chainer
    class Graph
      attr_reader :nodes, :inputs

      class << self
        def parse(onnx_graph)
          nodes = onnx_graph.node
          initializers = onnx_graph.initializer
          outputs = onnx_graph.output

          # take out input
          initializer_names = onnx_graph.initializer.map(&:name)
          call_inputs = onnx_graph.input.reject { |i| initializer_names.include?(i.name) }
          name = 'x'
          @inputs = call_inputs.each_with_object({}) do |i, hash|
            hash[i.name] = name
            name = name.succ
          end

          # parse each node
          output_name_index = {}
          @nodes = nodes.map do |n|
            output_name_index[n.op_type] ||= 1
            klass = operator_klass(n.op_type)
            node = klass.parse(n, initializers.map(&:name), onnx_graph.input, output_name_index[n.op_type])
            output_name_index[n.op_type] += 1
            node
          end

          # take out output
          output_names = {}
          @nodes.each { |n| output_names.merge!(n.output_names) }


          puts "initlize"
          @nodes.select(&:need_initialized).each do |n|
            puts n.to_initialize_string
          end

          # output nodes
          puts "call"
          @nodes.each do |n|
            args = n.input_names.map { |name| @inputs[name] || output_names[name] }
            puts n.to_call_string(args)
          end

        end

        private

        def operator_klass(op_type)
          case op_type
          when 'Gemm' then
            return Onnx::Chainer::Operators::Gemm
          when 'Relu' then
            return Onnx::Chainer::Operators::Relu
          end
        end
      end

      # export file
      def export
        binding.irb
      end
    end
  end
end
