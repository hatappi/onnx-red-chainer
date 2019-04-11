module OnnxChainer
  class Operator
    attr_reader :need_initialized,
                :output_names,
                :input_names

    class << self
      def parse(node)
      end
    end

    def chainer_class
      raise NotImplementedError
    end

    def to_initialize_string
      raise NotImplementedError
    end

    def to_call_string
      raise NotImplementedError
    end
  end
end
