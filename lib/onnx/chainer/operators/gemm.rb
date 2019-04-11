require 'onnx/chainer/operator'

module Onnx
  module Chainer
    module Operators
      class Gemm < Operator
        class << self
          def parse(node, input_names, inputs, output_name_index)
            bias_name = node.input.find { |i| i.match(/_b$/) }
            input = inputs.find { |i| i.name == bias_name }
            output_shape = input.type.tensor_type.shape.dim.map(&:dim_value)

            need_initialized = node.input.any? { |i| inputs.map(&:name).include?(i) }

            output_names = {
              node.output.first => "h#{output_name_index}"
            }
            instance_variable_name = "@h#{output_name_index}"

            self.new(input_names: input_names, output_shape: output_shape, output_names: output_names, instance_variable_name: instance_variable_name, need_initialized: need_initialized)
          end
        end

        def initialize(input_names:, output_shape:, output_names:, instance_variable_name:, need_initialized:)
          @input_names = input_names
          @output_shape = output_shape
          @output_names = output_names
          @instance_variable_name = instance_variable_name
          @need_initialized = need_initialized
        end

        def chainer_class
          ::Chainer::Links::Connection::Linear
        end

        def to_initialize_string
          "#{@instance_variable_name} = #{chainer_class}.new(nil, output_size: #{@output_shape})"
        end

        def to_call_string(args)
          "#{@output_names.values.first} = #{@instance_variable_name}.(#{args.join(', ')})"
        end
      end
    end
  end
end
