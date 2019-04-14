# ONNX-Red-Chainer
This is an add-on package for ONNX support by Red Chainer.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'onnx-red-chainer'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install onnx-red-chainer

## Usage

```
$ bundle exec onnx-red-chainer -h
onnx-red-chainer [OPTIONS] FILE
    -o, --output_dir=OUTPUT_DIR      output path
    -m, --model_name=MODEL_NAME      Model name
    -h, --help                       show help
```

### Run Test

```
$ bundle exec ruby test/run_test.rb
```

## Supported Functions
### Activation
- ReLU

### Connection
- LinearFunction

## Development


## License
The MIT license.  
See [LICENSE.txt](./LICENSE.txt) for details.