import os,sys,time,json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import paths
from model import Inference
from prompts import Prompts
from examples import Examples

# ------------ check command line and get arguments -----------------
def get_arguments():
    # [MOD-1.3] Fix: original code accessed sys.argv[6] unconditionally, crashing
    # with IndexError when the optional -quant/-ollama flag was omitted (the
    # default sbatch call path, since fewshot.sh forwards $QUANT which is empty
    # when unset). Per professor's 2026-04-10 notice about fixing parameter
    # parsing in fewshot / finetune-train / finetune-inference.
    # [MOD-1.3] also accepts an optional -balanced flag (Phase G) in any
    # position after position 5 to request class-balanced FS sampling.
    argv = sys.argv[:]
    balanced = False
    if "-balanced" in argv:
        balanced = True
        argv = [a for a in argv if a != "-balanced"]

    if not 6 <= len(argv) <= 7:
        print(f"Usage:  {sys.argv[0]} model prompts num_few_shot trainfile testfile [(-quant|-ollama)] [-balanced]", file=sys.stderr)
        sys.exit(1)
    if len(argv) == 7 and argv[6] not in ["-quant", "-ollama"]:
        print(f"Usage:  {sys.argv[0]} model prompts num_few_shot trainfile testfile [(-quant|-ollama)] [-balanced]", file=sys.stderr)
        sys.exit(1)

    model = argv[1]
    promptfile = argv[2]
    num_few_shot = int(argv[3])
    traindata = argv[4]
    testdata = argv[5]
    flag = argv[6] if len(argv) == 7 else ""
    quantized = (flag == "-quant")
    ollama = (flag == "-ollama")

    return model, promptfile, num_few_shot, traindata, testdata, quantized, ollama, balanced


############## main ###################

# get command line arguments
model, promptfile, num_few_shot, traindata, testdata, quantized, ollama, balanced = get_arguments()

print(f"========= FEW SHOT === PROMPTS={promptfile}  SHOTS={num_few_shot}  DATA={testdata} quantized={quantized} balanced={balanced}", file=sys.stderr)

# load training data (FS examples)
trainfile = os.path.join(paths.DATA,traindata+".xml")
fs_examples = Examples(trainfile, "NER").select_examples(num_few_shot, balanced=balanced)

# load prompts, create few-shot prompt
prompts = Prompts(promptfile, fs_examples)

# load test data
testfile = os.path.join(paths.DATA,testdata+".xml")
test = Examples(testfile, "NER")

# load model and tokenizer
t0 = time.time()
if ollama:
   engine = Inference(model, ollama=True)
else :
   # [MOD-1.3] Path updated per professor's 2026-04-10 notice: mml0 -> mgl0
   MODEL_PATH = f"/scratch/nas/1/PDI/mgl0/models/{model}"
   engine = Inference(MODEL_PATH, quantized=quantized)
print(f"Model loading took {time.time()-t0:.1f} seconds", file=sys.stderr)

# annotate each example in testdata
t0 = time.time()
annotated = []
for i,ex in enumerate(test.select_examples()):
    print(f"Processing example {i} - {ex['id']}", flush=True, file=sys.stderr)
    
    # create prompt for this example, adding it to FS prompt
    messages = prompts.prepare_messages(ex['input'])
    # call model to generate response 
    gen_text = engine.generate(messages)
    # store responses
    ex['predicted'] = gen_text
    ex['evaluator'] = test.eval_format(ex,gen_text)
    annotated.append(ex)

print("Done", file=sys.stderr)
print(f"Processed {len(annotated)} examples in {time.time()-t0:.1f} seconds. ({(time.time()-t0)/len(annotated):.2f} sec/example)", file=sys.stderr)

os.makedirs(paths.RESULTS, exist_ok=True)
quant = "-quant" if quantized else ""
outfname = os.path.join(paths.RESULTS,
                        f"FS-{model}-{num_few_shot}-{testdata}{quant}")
with open(outfname+".json", "w") as of:  
   json.dump(annotated, of, indent=1, ensure_ascii=False)
with open(outfname+".out", "w") as of:  
   for e in annotated:
      if e["evaluator"]: 
          print("\n".join(e["evaluator"]), file=of)

# clean up gpu
del engine
torch.cuda.empty_cache() 

