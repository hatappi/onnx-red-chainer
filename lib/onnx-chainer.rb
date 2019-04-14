require 'chainer'
require "onnx-chainer/version"
require "onnx-chainer/graph"

org_verbose = $VERBOSE
$VERBOSE = nil
require "onnx-chainer/proto/onnx_pb"
$VERBOSE = org_verbose

module OnnxChainer
  class Error < StandardError; end

  def self.parse_file(onnx_path)
    raise "File not found. #{onnx_path}" if onnx_path.nil? || !File.exists?(onnx_path)

    m = Onnx::ModelProto.decode(File.read(onnx_path))

    OnnxChainer::Graph.parse(m.graph)
  end
end
