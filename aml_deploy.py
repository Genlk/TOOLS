import os
import shutil
import paramiko
from export_onnx import main as export_onnx_fun
from nanodet.util import cfg, load_config

# Flags
FLAG_TRANS_CKPT_TO_ONNX = True
FLAG_TRANS_ONNX_TO_AML = True
FLAG_NOT_QUANT = False
FLAG_QUANT_8_BIT = False
FLAG_QUANT_HYPER = True
FLAG_QUANT_HYPER_MODEL = "backbone" # backbone_fpn   backbone
FLAG_GENERATE_CASE_CODE = True
FLAG_ANDROID_BUILD = True
FLAG_SCP_TO_LOCAL = False
FLAG_UPLOAD_TO_ANDROID = False

#GLOBAL_VALUES
ANDROID_TOOLS_DIR = "/home/kai/project/C308_tools/hah/DDK_6.4.4.3_SDK_V0.1/android_sdk_6443/android_sdk_6443"
AML_TOOLS_DIR = "/home/kai/project/C308_tools/DDK_6.4.4.3_Tool_acuity-toolkit-binary-5.16.3/acuity-toolkit-binary-5.16.3"
HOST_IP = "10.64.30.201"
NDK_ROOT_DIR = "/home/kai/project/C308_tools/android-ndk-r17/ndk-build"

#VARIEABLES
MODEL_TAG = "nano_0222_1"
CFG_PATH = "config/nanodet-plus_384_with_zhongbao.yml"
MODEL_PATH = "/data/disk1/kai/nanodet/nanodet-plus-384-with-zhongbao/NanoDet/2023-02-22-13-50-10/checkpoints/sample-mnist-epoch=214-train_loss=2.30.ckpt"
LOCAL_PATH = ""
BEFORE_MODEL_POOLING = True

def trans_ckpt_to_onnx(cfg_path=None, model_path=None, input_shape=None):
    if not FLAG_TRANS_CKPT_TO_ONNX:
        return 0
    if cfg_path is None:
        cfg_path = CFG_PATH
    if model_path is None:
        model_path = MODEL_PATH
    os.chdir("/home/kai/project/nanodet/")
    load_config(cfg, cfg_path)
    if input_shape is None:
        input_shape = cfg.data.train.input_size
        if BEFORE_MODEL_POOLING:
            input_shape = [2 * e for e in input_shape]
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    out_path_dir = os.path.join(AML_TOOLS_DIR, MODEL_TAG)
    if not os.path.exists(out_path_dir):
        os.mkdir(out_path_dir)
    out_path = os.path.join(out_path_dir, "nanodet_model_best.onnx")
    export_onnx_fun(cfg, model_path, out_path, input_shape)
    print("Model saved to:", out_path)
    return 0

def trans_onnx_to_aml():
    if not FLAG_TRANS_ONNX_TO_AML:
        return 0
    os.chdir(AML_TOOLS_DIR)
    trans_cmd = "./bin/convertonnx --onnx-model ./%s/nanodet_model_best.onnx --net-output ./%s/nanodet.json --data-output ./%s/nanodet.data"%(MODEL_TAG, MODEL_TAG, MODEL_TAG)
    trans_process = os.popen(trans_cmd)
    while True:
        line = trans_process.read()
        if line == '':
            break
    if os.path.exists("./%s/nanodet.json"%MODEL_TAG) and os.path.exists("./%s/nanodet.data"%MODEL_TAG):
        print("data file and json file already finished!")
    else:
        print("data file and json file not finished!!! Please try again!")
        return 1
    return 0

def quant_model_8bit():
    quant_cmd = "./bin/tensorzonex --action quantization --quantized-dtype asymmetric_affine-u8 --channel-mean-value '113.533554 118.14172 123.63607 21.405144' --source text --source-file dataset_zicai.txt --model-input ./%s/nanodet.json --model-data ./%s/nanodet.data --reorder-channel '2 1 0' --quantized-rebuild --batch-size 100 --epochs 10"%(MODEL_TAG, MODEL_TAG)
    os.chdir(AML_TOOLS_DIR)
    quant_process = os.popen(quant_cmd)
    while True:
        line = quant_process.read()
        if line == '':
            break
    if os.path.exists("./%s/nanodet.quantize" % MODEL_TAG):
        print(".quantize file  already finished!")
    else:
        print(".quantize file   not finished!!! Please try again!")
        return 1
    print("8 bit quant finished!")

def rewrite_quantfile_for_hyper():
    if FLAG_QUANT_HYPER_MODEL == "backbone":
        layers = "{attach_Add_/backbone/stage2/stage2.2/Add/out0_2: dynamic_fixed_point-i16, attach_Add_/backbone/stage4/stage4.2/Add/out0_3: dynamic_fixed_point-i16, attach_LeakyRelu_/backbone/stage6/stage6.1/stage6.1.2/LeakyRelu/out0_4: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage6/stage6.1/stage6.1.2/LeakyRelu_13: dynamic_fixed_point-i16, Add_/backbone/stage4/stage4.2/Add_14: dynamic_fixed_point-i16, Add_/backbone/stage2/stage2.2/Add_15: dynamic_fixed_point-i16, Conv_/backbone/stage6/stage6.1/stage6.1.0/Conv_26: dynamic_fixed_point-i16, Add_/backbone/stage4/stage4.1/Add_27: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.2/conv/conv.2/Conv_28: dynamic_fixed_point-i16, Add_/backbone/stage2/stage2.1/Add_29: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.2/conv/conv.2/Conv_30: dynamic_fixed_point-i16, Conv_/backbone/stage6/stage6.0/conv/conv.2/Conv_44: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.0/conv/conv.2/Conv_45: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.1/conv/conv.2/Conv_46: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.2/conv/conv.1/conv.1.2/LeakyRelu_47: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.0/conv/conv.2/Conv_48: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.1/conv/conv.2/Conv_49: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.2/conv/conv.1/conv.1.2/LeakyRelu_50: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage6/stage6.0/conv/conv.1/conv.1.2/LeakyRelu_67: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.0/conv/conv.1/conv.1.2/LeakyRelu_68: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.1/conv/conv.1/conv.1.2/LeakyRelu_69: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.2/conv/conv.1/conv.1.0/Conv_70: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.0/conv/conv.1/conv.1.2/LeakyRelu_71: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.1/conv/conv.1/conv.1.2/LeakyRelu_72: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.2/conv/conv.1/conv.1.0/Conv_73: dynamic_fixed_point-i16, Conv_/backbone/stage6/stage6.0/conv/conv.1/conv.1.0/Conv_93: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.0/conv/conv.1/conv.1.0/Conv_94: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.1/conv/conv.1/conv.1.0/Conv_95: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.2/conv/conv.0/conv.0.2/LeakyRelu_96: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.0/conv/conv.1/conv.1.0/Conv_97: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.1/conv/conv.1/conv.1.0/Conv_98: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.2/conv/conv.0/conv.0.2/LeakyRelu_99: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.2/conv/conv.0/conv.0.0/Conv_100: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage6/stage6.0/conv/conv.0/conv.0.2/LeakyRelu_120: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.0/conv/conv.0/conv.0.2/LeakyRelu_121: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.1/conv/conv.0/conv.0.2/LeakyRelu_122: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.2/conv/conv.0/conv.0.0/Conv_123: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.0/conv/conv.0/conv.0.2/LeakyRelu_124: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.1/conv/conv.0/conv.0.2/LeakyRelu_125: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.1/conv/conv.0/conv.0.0/Conv_126: dynamic_fixed_point-i16, Conv_/backbone/stage6/stage6.0/conv/conv.0/conv.0.0/Conv_138: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.0/conv/conv.0/conv.0.0/Conv_139: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.1/conv/conv.0/conv.0.0/Conv_140: dynamic_fixed_point-i16, Add_/backbone/stage3/stage3.3/Add_141: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.0/conv/conv.0/conv.0.0/Conv_142: dynamic_fixed_point-i16, Add_/backbone/stage1/stage1.1/Add_143: dynamic_fixed_point-i16, Add_/backbone/stage5/stage5.2/Add_154: dynamic_fixed_point-i16, Add_/backbone/stage3/stage3.2/Add_155: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.3/conv/conv.2/Conv_156: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.0/conv/conv.2/Conv_157: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.1/conv/conv.2/Conv_158: dynamic_fixed_point-i16, Add_/backbone/stage5/stage5.1/Add_166: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.2/conv/conv.2/Conv_167: dynamic_fixed_point-i16, Add_/backbone/stage3/stage3.1/Add_168: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.2/conv/conv.2/Conv_169: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.3/conv/conv.1/conv.1.2/LeakyRelu_170: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage1/stage1.0/conv/conv.1/conv.1.2/LeakyRelu_171: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage1/stage1.1/conv/conv.1/conv.1.2/LeakyRelu_172: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.0/conv/conv.2/Conv_177: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.1/conv/conv.2/Conv_178: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.2/conv/conv.1/conv.1.2/LeakyRelu_179: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.0/conv/conv.2/Conv_180: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.1/conv/conv.2/Conv_181: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.2/conv/conv.1/conv.1.2/LeakyRelu_182: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.3/conv/conv.1/conv.1.0/Conv_183: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.0/conv/conv.1/conv.1.0/Conv_184: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.1/conv/conv.1/conv.1.0/Conv_185: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.0/conv/conv.1/conv.1.2/LeakyRelu_189: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.1/conv/conv.1/conv.1.2/LeakyRelu_190: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.2/conv/conv.1/conv.1.0/Conv_191: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.0/conv/conv.1/conv.1.2/LeakyRelu_192: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.1/conv/conv.1/conv.1.2/LeakyRelu_193: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.2/conv/conv.1/conv.1.0/Conv_194: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.3/conv/conv.0/conv.0.2/LeakyRelu_195: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage1/stage1.0/conv/conv.0/conv.0.2/LeakyRelu_196: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage1/stage1.1/conv/conv.0/conv.0.2/LeakyRelu_197: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.0/conv/conv.1/conv.1.0/Conv_200: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.1/conv/conv.1/conv.1.0/Conv_201: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.2/conv/conv.0/conv.0.2/LeakyRelu_202: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.0/conv/conv.1/conv.1.0/Conv_203: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.1/conv/conv.1/conv.1.0/Conv_204: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.2/conv/conv.0/conv.0.2/LeakyRelu_205: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.3/conv/conv.0/conv.0.0/Conv_206: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.0/conv/conv.0/conv.0.0/Conv_207: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.1/conv/conv.0/conv.0.0/Conv_208: dynamic_fixed_point-i16, Conv_/backbone/stage0/stage0.0/conv/conv.1/Conv_210: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.0/conv/conv.0/conv.0.2/LeakyRelu_211: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.1/conv/conv.0/conv.0.2/LeakyRelu_212: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.2/conv/conv.0/conv.0.0/Conv_213: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.0/conv/conv.0/conv.0.2/LeakyRelu_214: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.1/conv/conv.0/conv.0.2/LeakyRelu_215: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.2/conv/conv.0/conv.0.0/Conv_216: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.1/conv/conv.0/conv.0.0/Conv_217: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage0/stage0.0/conv/conv.0/conv.0.2/LeakyRelu_218: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.0/conv/conv.0/conv.0.0/Conv_219: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.1/conv/conv.0/conv.0.0/Conv_220: dynamic_fixed_point-i16, Conv_/backbone/stage0/stage0.0/conv/conv.0/conv.0.0/Conv_221: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.0/conv/conv.0/conv.0.0/Conv_222: dynamic_fixed_point-i16, LeakyRelu_/backbone/first_layer/first_layer.2/LeakyRelu_223: dynamic_fixed_point-i16, Conv_/backbone/first_layer/first_layer.0/Conv_224: dynamic_fixed_point-i16}"
    elif FLAG_QUANT_HYPER_MODEL == "backbone_fpn":
        layers = "{attach_Add_/backbone/stage2/stage2.2/Add/out0_2: dynamic_fixed_point-i16, attach_Add_/backbone/stage4/stage4.2/Add/out0_3: dynamic_fixed_point-i16, attach_LeakyRelu_/backbone/stage6/stage6.1/stage6.1.2/LeakyRelu/out0_4: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage6/stage6.1/stage6.1.2/LeakyRelu_13: dynamic_fixed_point-i16, Add_/backbone/stage4/stage4.2/Add_14: dynamic_fixed_point-i16, Add_/backbone/stage2/stage2.2/Add_15: dynamic_fixed_point-i16, Conv_/backbone/stage6/stage6.1/stage6.1.0/Conv_26: dynamic_fixed_point-i16, Add_/backbone/stage4/stage4.1/Add_27: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.2/conv/conv.2/Conv_28: dynamic_fixed_point-i16, Add_/backbone/stage2/stage2.1/Add_29: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.2/conv/conv.2/Conv_30: dynamic_fixed_point-i16, Conv_/backbone/stage6/stage6.0/conv/conv.2/Conv_44: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.0/conv/conv.2/Conv_45: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.1/conv/conv.2/Conv_46: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.2/conv/conv.1/conv.1.2/LeakyRelu_47: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.0/conv/conv.2/Conv_48: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.1/conv/conv.2/Conv_49: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.2/conv/conv.1/conv.1.2/LeakyRelu_50: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage6/stage6.0/conv/conv.1/conv.1.2/LeakyRelu_67: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.0/conv/conv.1/conv.1.2/LeakyRelu_68: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.1/conv/conv.1/conv.1.2/LeakyRelu_69: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.2/conv/conv.1/conv.1.0/Conv_70: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.0/conv/conv.1/conv.1.2/LeakyRelu_71: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.1/conv/conv.1/conv.1.2/LeakyRelu_72: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.2/conv/conv.1/conv.1.0/Conv_73: dynamic_fixed_point-i16, Conv_/backbone/stage6/stage6.0/conv/conv.1/conv.1.0/Conv_93: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.0/conv/conv.1/conv.1.0/Conv_94: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.1/conv/conv.1/conv.1.0/Conv_95: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.2/conv/conv.0/conv.0.2/LeakyRelu_96: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.0/conv/conv.1/conv.1.0/Conv_97: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.1/conv/conv.1/conv.1.0/Conv_98: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.2/conv/conv.0/conv.0.2/LeakyRelu_99: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.2/conv/conv.0/conv.0.0/Conv_100: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage6/stage6.0/conv/conv.0/conv.0.2/LeakyRelu_120: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.0/conv/conv.0/conv.0.2/LeakyRelu_121: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage4/stage4.1/conv/conv.0/conv.0.2/LeakyRelu_122: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.2/conv/conv.0/conv.0.0/Conv_123: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.0/conv/conv.0/conv.0.2/LeakyRelu_124: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage2/stage2.1/conv/conv.0/conv.0.2/LeakyRelu_125: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.1/conv/conv.0/conv.0.0/Conv_126: dynamic_fixed_point-i16, Conv_/backbone/stage6/stage6.0/conv/conv.0/conv.0.0/Conv_138: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.0/conv/conv.0/conv.0.0/Conv_139: dynamic_fixed_point-i16, Conv_/backbone/stage4/stage4.1/conv/conv.0/conv.0.0/Conv_140: dynamic_fixed_point-i16, Add_/backbone/stage3/stage3.3/Add_141: dynamic_fixed_point-i16, Conv_/backbone/stage2/stage2.0/conv/conv.0/conv.0.0/Conv_142: dynamic_fixed_point-i16, Add_/backbone/stage1/stage1.1/Add_143: dynamic_fixed_point-i16, Add_/backbone/stage5/stage5.2/Add_154: dynamic_fixed_point-i16, Add_/backbone/stage3/stage3.2/Add_155: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.3/conv/conv.2/Conv_156: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.0/conv/conv.2/Conv_157: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.1/conv/conv.2/Conv_158: dynamic_fixed_point-i16, Add_/backbone/stage5/stage5.1/Add_166: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.2/conv/conv.2/Conv_167: dynamic_fixed_point-i16, Add_/backbone/stage3/stage3.1/Add_168: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.2/conv/conv.2/Conv_169: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.3/conv/conv.1/conv.1.2/LeakyRelu_170: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage1/stage1.0/conv/conv.1/conv.1.2/LeakyRelu_171: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage1/stage1.1/conv/conv.1/conv.1.2/LeakyRelu_172: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.0/conv/conv.2/Conv_177: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.1/conv/conv.2/Conv_178: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.2/conv/conv.1/conv.1.2/LeakyRelu_179: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.0/conv/conv.2/Conv_180: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.1/conv/conv.2/Conv_181: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.2/conv/conv.1/conv.1.2/LeakyRelu_182: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.3/conv/conv.1/conv.1.0/Conv_183: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.0/conv/conv.1/conv.1.0/Conv_184: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.1/conv/conv.1/conv.1.0/Conv_185: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.0/conv/conv.1/conv.1.2/LeakyRelu_189: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.1/conv/conv.1/conv.1.2/LeakyRelu_190: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.2/conv/conv.1/conv.1.0/Conv_191: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.0/conv/conv.1/conv.1.2/LeakyRelu_192: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.1/conv/conv.1/conv.1.2/LeakyRelu_193: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.2/conv/conv.1/conv.1.0/Conv_194: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.3/conv/conv.0/conv.0.2/LeakyRelu_195: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage1/stage1.0/conv/conv.0/conv.0.2/LeakyRelu_196: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage1/stage1.1/conv/conv.0/conv.0.2/LeakyRelu_197: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.0/conv/conv.1/conv.1.0/Conv_200: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.1/conv/conv.1/conv.1.0/Conv_201: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.2/conv/conv.0/conv.0.2/LeakyRelu_202: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.0/conv/conv.1/conv.1.0/Conv_203: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.1/conv/conv.1/conv.1.0/Conv_204: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.2/conv/conv.0/conv.0.2/LeakyRelu_205: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.3/conv/conv.0/conv.0.0/Conv_206: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.0/conv/conv.0/conv.0.0/Conv_207: dynamic_fixed_point-i16, Conv_/backbone/stage1/stage1.1/conv/conv.0/conv.0.0/Conv_208: dynamic_fixed_point-i16, Conv_/backbone/stage0/stage0.0/conv/conv.1/Conv_210: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.0/conv/conv.0/conv.0.2/LeakyRelu_211: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage5/stage5.1/conv/conv.0/conv.0.2/LeakyRelu_212: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.2/conv/conv.0/conv.0.0/Conv_213: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.0/conv/conv.0/conv.0.2/LeakyRelu_214: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage3/stage3.1/conv/conv.0/conv.0.2/LeakyRelu_215: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.2/conv/conv.0/conv.0.0/Conv_216: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.1/conv/conv.0/conv.0.0/Conv_217: dynamic_fixed_point-i16, LeakyRelu_/backbone/stage0/stage0.0/conv/conv.0/conv.0.2/LeakyRelu_218: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.0/conv/conv.0/conv.0.0/Conv_219: dynamic_fixed_point-i16, Conv_/backbone/stage5/stage5.1/conv/conv.0/conv.0.0/Conv_220: dynamic_fixed_point-i16, Conv_/backbone/stage0/stage0.0/conv/conv.0/conv.0.0/Conv_221: dynamic_fixed_point-i16, Conv_/backbone/stage3/stage3.0/conv/conv.0/conv.0.0/Conv_222: dynamic_fixed_point-i16, LeakyRelu_/backbone/first_layer/first_layer.2/LeakyRelu_223: dynamic_fixed_point-i16, Conv_/backbone/first_layer/first_layer.0/Conv_224: dynamic_fixed_point-i16, attach_Transpose_/head/Transpose/out0_0: dynamic_fixed_point-i16, attach_Sigmoid_/head_cls/Sigmoid/out0_1: dynamic_fixed_point-i16, Sigmoid_/head_cls/Sigmoid_16: dynamic_fixed_point-i16, Transpose_/head/Transpose_17: dynamic_fixed_point-i16, Gemm_/head_cls/linear_relu_stack_straight/linear_relu_stack_straight.0/Gemm_31: dynamic_fixed_point-i16, Concat_/head/Concat_4_32: dynamic_fixed_point-i16, Flatten_/head_cls/flatten/Flatten_51: dynamic_fixed_point-i16, Reshape_/head/Reshape_52: dynamic_fixed_point-i16, Reshape_/head/Reshape_1_53: dynamic_fixed_point-i16, Reshape_/head/Reshape_2_54: dynamic_fixed_point-i16, Reshape_/head/Reshape_3_55: dynamic_fixed_point-i16, GlobalAveragePool_/head_cls/pool/GlobalAveragePool_74: dynamic_fixed_point-i16, Conv_/head/gfl_cls.0/Conv_75: dynamic_fixed_point-i16, Conv_/head/gfl_cls.1/Conv_76: dynamic_fixed_point-i16, Conv_/head/gfl_cls.2/Conv_77: dynamic_fixed_point-i16, Conv_/head/gfl_cls.3/Conv_78: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.0.1/act_1/LeakyRelu_101: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.1.1/act_1/LeakyRelu_102: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.2.1/act_1/LeakyRelu_103: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.3.1/act_1/LeakyRelu_104: dynamic_fixed_point-i16, Conv_/head/cls_convs.3.1/pointwise/Conv_105: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.3.1/act/LeakyRelu_106: dynamic_fixed_point-i16, Conv_/head/cls_convs.3.1/depthwise/Conv_108: dynamic_fixed_point-i16, Conv_/head/cls_convs.0.1/pointwise/Conv_127: dynamic_fixed_point-i16, Conv_/head/cls_convs.1.1/pointwise/Conv_128: dynamic_fixed_point-i16, Conv_/head/cls_convs.2.1/pointwise/Conv_129: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.3.0/act_1/LeakyRelu_130: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.0.1/act/LeakyRelu_144: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.1.1/act/LeakyRelu_145: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.2.1/act/LeakyRelu_146: dynamic_fixed_point-i16, Conv_/head/cls_convs.3.0/pointwise/Conv_147: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.3.0/act/LeakyRelu_148: dynamic_fixed_point-i16, Conv_/head/cls_convs.0.1/depthwise/Conv_159: dynamic_fixed_point-i16, Conv_/head/cls_convs.1.1/depthwise/Conv_160: dynamic_fixed_point-i16, Conv_/head/cls_convs.2.1/depthwise/Conv_161: dynamic_fixed_point-i16, Conv_/head/cls_convs.3.0/depthwise/Conv_162: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.2.0/act_1/LeakyRelu_163: dynamic_fixed_point-i16, Conv_/head/cls_convs.2.0/pointwise/Conv_164: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.0.0/act_1/LeakyRelu_173: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.1.0/act_1/LeakyRelu_174: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.2.0/act/LeakyRelu_175: dynamic_fixed_point-i16, Conv_/head/cls_convs.2.0/depthwise/Conv_176: dynamic_fixed_point-i16, Conv_/head/cls_convs.0.0/pointwise/Conv_186: dynamic_fixed_point-i16, Conv_/head/cls_convs.1.0/pointwise/Conv_187: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.1.0/act/LeakyRelu_188: dynamic_fixed_point-i16, LeakyRelu_/head/cls_convs.0.0/act/LeakyRelu_198: dynamic_fixed_point-i16, Conv_/head/cls_convs.1.0/depthwise/Conv_199: dynamic_fixed_point-i16, Conv_/head/cls_convs.0.0/depthwise/Conv_209: dynamic_fixed_point-i16, Flatten_/head_cls/flatten/Flatten_51_acuity_mark_perm_226: dynamic_fixed_point-i16, Reshape_/head/Reshape_52_acuity_mark_perm_227: dynamic_fixed_point-i16, Reshape_/head/Reshape_1_53_acuity_mark_perm_228: dynamic_fixed_point-i16, Reshape_/head/Reshape_2_54_acuity_mark_perm_229: dynamic_fixed_point-i16, Reshape_/head/Reshape_3_55_acuity_mark_perm_230: dynamic_fixed_point-i16}"
    else:
        print("Error in rewrite_quantfile")
    quant_file = "./%s/nanodet.quantize"%(MODEL_TAG)
    os.chdir(AML_TOOLS_DIR)
    with open(quant_file,"r",encoding="utf-8") as f:
        lines = f.readlines()
    lines[-1] = lines[-1].replace("{}",layers)
    with open(quant_file,"w",encoding="utf-8") as f:
        f.writelines(lines)

def quant_model_hyper():
    quant_cmd = "./bin/tensorzonex --action quantization --dtype float32 --quantized-dtype asymmetric_affine-u8 --channel-mean-value '113.533554 118.14172 123.63607 21.405144' --source text --source-file dataset_zicai.txt --model-input ./%s/nanodet.json --model-data ./%s/nanodet.data --reorder-channel '2 1 0' --quantized-rebuild --batch-size 100 --epochs 10"%(MODEL_TAG, MODEL_TAG)
    os.chdir(AML_TOOLS_DIR)
    quant_process = os.popen(quant_cmd)
    while True:
        line = quant_process.read()
        if line == '':
            break
    if os.path.exists("./%s/nanodet.quantize" % MODEL_TAG):
        print(".quantize file  already finished!")
    else:
        print(".quantize file   not finished!!! Please try again!")
        return 1
    print("Hyper quant 1/2 step finished!")
    rewrite_quantfile_for_hyper()
    os.chdir(AML_TOOLS_DIR)
    hyper_quant_cmd = "./bin/tensorzonex --action quantization --dtype float32 --quantized-dtype asymmetric_affine-u8 --channel-mean-value '113.533554 118.14172 123.63607 21.405144' --source text --source-file dataset_zicai.txt --model-input ./%s/nanodet.json --model-data ./%s/nanodet.data --reorder-channel '2 1 0' --batch-size 100 --epochs 10 --quantized-hybrid --model-quantize ./%s/nanodet.quantize"%(MODEL_TAG, MODEL_TAG, MODEL_TAG)
    hyper_quant_process = os.popen(hyper_quant_cmd)
    while True:
        line = hyper_quant_process.read()
        if line == '':
            break
    print("Hyper quant 2/2 step finished!")

def quant_model():
    if FLAG_NOT_QUANT:
        print("Not need quant!")
        return 0
    if not FLAG_QUANT_8_BIT^FLAG_QUANT_HYPER:
        print("Only one quant type be choice!")
        return 0
    if FLAG_QUANT_8_BIT:
        quant_model_8bit()
    elif FLAG_QUANT_HYPER:
        quant_model_hyper()
    else:
        print("Error in quant files!")

def generate_case_code():
    if not FLAG_GENERATE_CASE_CODE:
        return 0
    os.chdir(os.path.join(AML_TOOLS_DIR, MODEL_TAG))
    generate_cmd = "../bin/ovxgenerator --model-input ./nanodet.quantize.json --data-input ./nanodet.data  --model-quantize ./nanodet.quantize --export-dtype quantized --channel-mean-value '113.533554 118.14172 123.63607 21.405144' --optimize VIPNANOQI_PID0X88 --reorder-channel '2 1 0' --viv-sdk ../bin/vcmdtools --pack-nbg-unify"
    if FLAG_QUANT_8_BIT:
        generate_cmd += " --model-input ./nanodet.json"
    elif FLAG_QUANT_HYPER:
        generate_cmd += " --model-input ./nanodet.quantize.json --model-quantize ./nanodet.quantize"
    case_process = os.popen(generate_cmd)
    while True:
        line = case_process.read()
        if line == '':
            break
    print("Generate case code finished!")

def copy_build_files(src_dir, dst_dir):
    for build_file in os.listdir(src_dir):
        shutil.copyfile(
            os.path.join(src_dir,build_file),
            os.path.join(dst_dir,build_file)
        )

def android_build():
    if not FLAG_ANDROID_BUILD:
        return 0
    os.chdir(ANDROID_TOOLS_DIR)
    project_dir = os.path.join(ANDROID_TOOLS_DIR, MODEL_TAG)
    print("Project dir is %s \n" %project_dir)
    print("Project dir is %s \n" % project_dir)
    if not os.path.exists(project_dir):
        os.mkdir(project_dir)
    else:
        shutil.rmtree(project_dir)
        os.mkdir(project_dir)
    shutil.copytree(
        os.path.join(AML_TOOLS_DIR,MODEL_TAG+"_nbg_unify"),
        os.path.join(ANDROID_TOOLS_DIR,MODEL_TAG,"jni/")
    )
    if FLAG_QUANT_8_BIT:
        copy_build_files(
            os.path.join(ANDROID_TOOLS_DIR, "nano_768", "8_bit"),
            os.path.join(ANDROID_TOOLS_DIR, MODEL_TAG, "jni")
        )
    elif FLAG_QUANT_HYPER:
        copy_build_files(
            os.path.join(ANDROID_TOOLS_DIR, "nano_768", "hyper_bit"),
            os.path.join(ANDROID_TOOLS_DIR, MODEL_TAG, "jni")
        )
    else:
        print("Some thing error with the Quant type ,please set one of them true!")
    build_cmd = NDK_ROOT_DIR
    os.chdir(os.path.join(ANDROID_TOOLS_DIR, MODEL_TAG))
    build_process = os.popen(build_cmd)
    while True:
        line = build_process.read()
        if line == '':
            break
    print("Generate case code finished!")
    return 0

def scp_to_local():
    if not FLAG_SCP_TO_LOCAL:
        return 0
    src_dir = os.path.join(ANDROID_TOOLS_DIR, MODEL_TAG)
    dst_dir = r'D:\2-【code】\nanodet-main\nanodet-main\demo_aml'

    def remote_scp(remote_path, local_path, username="***", password="***"):
        t = paramiko.Transport((HOST_IP, 22))
        t.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(t)
        src = remote_path
        des = local_path
        sftp.get(src, des)
        t.close()
    remote_scp(src_dir,dst_dir)

def upload_to_board():
    #
    pass

if __name__ == "__main__":
    trans_ckpt_to_onnx()
    trans_onnx_to_aml()
    quant_model()
    generate_case_code()
    android_build()
    scp_to_local()
