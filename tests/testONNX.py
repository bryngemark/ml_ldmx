import onnx
m=onnx.load("outputs/tpad_transformer_2e_10k_h128_3layers_l4_lr3e4/export/model.onnx")
print("IR:", m.ir_version)
print("opsets:", [(x.domain or "ai.onnx", x.version) for x in m.opset_import])
