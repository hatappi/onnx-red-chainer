require 'onnx/chainer/operator'

module Onnx
  module Chainer
    module Operators
      class Relu < Operator
        class << self
          def parse(node, initializer_names, inputs, output_name_index)
            need_initialized = node.input.any? { |i| inputs.map(&:name).include?(i) }

            output_names = {
              node.output.first => "r#{output_name_index}"
            }
            instance_variable_name = "@r#{output_name_index}"

            input_names = node.input.reject { |i| initializer_names.include?(i) }

            self.new(input_names: input_names, output_names: output_names, instance_variable_name: instance_variable_name, need_initialized: need_initialized)
          end
        end

        def initialize(input_names:, output_names:, instance_variable_name:, need_initialized:)
          @input_names = input_names
          @output_names = output_names
          @instance_variable_name = instance_variable_name
          @need_initialized = need_initialized
        end

        def chainer_class
          ::Chainer::Functions::Activation::Relu
        end

        def to_call_string(args)
          "#{@output_names.values.first} = #{chainer_class}.relu(#{args.join(', ')})"
        end
      end
    end
  end
end
