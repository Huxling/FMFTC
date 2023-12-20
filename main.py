import warnings

warnings.filterwarnings("ignore")
from Run import pre_run
load_client_model, load_server_model = pre_run.train()
from Run import clu_run
clu_run.train(load_client_model=load_client_model,load_server_model=load_server_model)
