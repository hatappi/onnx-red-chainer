# frozen_string_literal: true

require 'onnx-chainer'
require 'json'
require 'optparse'
require 'pathname'

module OnnxChainer
  class CLI
    def self.start(argv)
      new(argv).run
    end

    def initialize(argv)
      @argv = argv.dup
      @parser = OptionParser.new do |opts|
        opts.banner = 'onnx-red-chainer [OPTIONS] FILE'
        opts.version = VERSION
        opts.on('-o', '--output_dir=OUTPUT_DIR', 'output path') { |v| @output = v }
        opts.on('-m', '--model_name=MODEL_NAME', 'Model name') { |v| @model_name = v }
        opts.on('-h', '--help', 'show help') { @help = true }
      end
      @onnx_path = argv.pop
      @parser.parse!(argv)
    end

    def run
      if @help || @argv.empty?
        puts @parser.help
      else
        graph = OnnxChainer.parse_file(@onnx_path)
        graph.export(output_dir: @output, model_name: @model_name)
      end
    end
  end
end
