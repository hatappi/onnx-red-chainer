
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "onnx-chainer/version"

Gem::Specification.new do |spec|
  spec.name          = "onnx-red-chainer"
  spec.version       = OnnxChainer::VERSION
  spec.authors       = ["hatappi"]
  spec.email         = ["hatappi@hatappi.me"]

  spec.summary       = "Automatically generate Ruby code from ONNX"
  spec.description   = "Automatically generate Ruby code from ONNX"
  spec.homepage      = "https://github.com/hatappi/onnx-red-chainer"
  spec.license       = "MIT"
  spec.files         = Dir.chdir(File.expand_path('..', __FILE__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency 'google-protobuf'
  spec.add_dependency 'red-chainer'
  spec.add_dependency "numo-narray"

  spec.add_development_dependency "bundler", "~> 1.17"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "test-unit", ">= 3.2.9"
end

