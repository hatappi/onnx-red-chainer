require 'onnx/chainer/operators/gemm'
require 'onnx/chainer/operators/relu'

module Onnx
  module Chainer
    class Graph
      attr_reader :nodes, :input_names, :output_names

      class << self
        def parse(onnx_graph)
          nodes = onnx_graph.node
          initializers = onnx_graph.initializer
          outputs = onnx_graph.output

          # take out input
          initializer_names = onnx_graph.initializer.map(&:name)
          call_inputs = onnx_graph.input.reject { |i| initializer_names.include?(i.name) }
          name = 'x'
          input_names = call_inputs.each_with_object({}) do |i, hash|
            hash[i.name] = name
            name = name.succ
          end

          # parse each node
          output_name_index = {}
          nodes = nodes.map do |n|
            output_name_index[n.op_type] ||= 1
            klass = operator_klass(n.op_type)
            i_names = n.input.reject { |i| initializers.map(&:name).include?(i) }

            node = klass.parse(n, i_names, onnx_graph.input, output_name_index[n.op_type])

            output_name_index[n.op_type] += 1
            node
          end

          # take out output
          output_names = {}
          nodes.each { |n| output_names.merge!(n.output_names) }

          self.new(onnx_graph.name, nodes, input_names, output_names)
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

      def initialize(model_name, nodes, input_names, output_names)
        @model_name = model_name
        @nodes = nodes
        @input_names = input_names
        @output_names = output_names
      end

      # export file
      def export(output_path: nil, model_name: nil)
        model_name = model_name || @model_name
        model_name = model_name.capitalize.gsub(/(?:^|_)(.)/){$1.upcase}

        output_path ||= 'model.rb'

s = <<EOS
require 'chainer'

class #{model_name} < Chainer::Chain
  def initialize()
    super()
    init_scope do
      #{@nodes.select(&:need_initialized).map(&:to_initialize_string).join("\n      ")}
    end
  end

  def call(#{@input_names.values.join(', ')})
    #{
      @nodes.map do |n|
        args = n.input_names.map { |name| @input_names[name] || @output_names[name] }
        n.to_call_string(args)
      end.join("\n    ")
    }
  end
end
EOS

        File.open(output_path, 'w') do |f|
          f.puts(s)
        end
      end
    end
  end
end
