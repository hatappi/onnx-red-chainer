require 'chainer'
require "onnx/chainer/version"
require "onnx/chainer/graph"
require "onnx/proto/onnx_pb"

module Onnx
  module Chainer
    class Error < StandardError; end

    def self.parse_file(onnx_path)
      m = Onnx::ModelProto.decode(File.read(onnx_path))

      puts "produced by #{m.producer_name}(#{m.producer_version})"

      Onnx::Chainer::Graph.parse(m.graph)
    end
  end
end
