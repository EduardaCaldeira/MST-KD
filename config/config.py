from easydict import EasyDict as edict

config = edict()

# ethnicity definition
config.ethnicity="All" # Asian | African | Caucasian | Indian | All

# teacher/student definition (change accordingly)
config.dataset = "balanced" # training dataset
config.batch_size = 128 # batch size per GPU
config.augmentation = "hf" # data augmentation policy
config.loss = "ElasticArcFace" #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus

# type of network to train [iresnet100 | iresnet50 | iresnet34]
config.network = "iresnet34"
config.SE = False # SEModule

if config.dataset == "balanced":
    config.rec = "/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_aligned" # directory for BalancedFace
    config.output = '/nas-ctm01/homes/mecaldeira/output/' + config.ethnicity + '/' + config.network + '/epochs_' + str(config.num_epoch)

    if config.ethnicity == "Indian":
        config.num_classes = 6997 # Indians have 6997 identities instead of 7000
    else:
        config.num_classes = 7000

    config.num_image = 324106 + 324202 + 274554 + 326070
    config.num_epoch = 52
    config.warmup_epoch = -1
    config.val_targets = ["African_test","Caucasian_test","Asian_test","Indian_test"]
    config.eval_step = 10000

    if config.num_epoch==52:
        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
                [m for m in [16, 28, 40, 50] if m - 1 <= epoch])
    else:
        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
                [m for m in [8, 14, 20, 25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

config.to_sample=28000 - 3 # 7000 | 28000 - 3: used for KD train + for training the baseline with 7k samples balanced across the 4 races

# adaptor parameters
config.middle_layer_size=512
config.is_dual_layer=True
config.has_drop_out=True
config.num_epoch_fc=26
config.lr_fc=1

# KD parameters
config.student_network="iresnet34"
config.loss_lambda=10000
config.arc_method="fusion" # selection | fusion | positioning | encoding

# generic baseline definitions
config.is_teacher_baseline=False
config.out_teacher_baseline='/nas-ctm01/homes/mecaldeira/output/teacher_baseline/'+config.ethnicity

# definitions for the baseline teachers' training
if config.is_teacher_baseline:
    config.rec="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline/teacher_"+config.ethnicity
    config.output=config.out_teacher_baseline + '/' + config.network + '/epochs_' + str(config.num_epoch)
    if config.ethnicity=="African":
        config.num_classes=7000
    else:
        config.num_classes=6999

# definitions for embedding extraction
if config.is_teacher_baseline:
    config.out_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline_embeddings/" + config.network + '/epochs_' + str(config.num_epoch)
    if config.network=="iresnet34": 
        if config.ethnicity=="Caucasian":
            config.best_model=41395
        elif config.ethnicity=="African":
            config.best_model=41871
        elif config.ethnicity=="Asian":
            config.best_model=31733
        else:
            config.best_model=38656
    elif config.network=="iresnet50":
        if config.ethnicity=="Caucasian":
            config.best_model=43830
        elif config.ethnicity=="African":
            config.best_model=44334
        elif config.ethnicity=="Asian":
            config.best_model=39056
        else:
            config.best_model=38656
    config.backbone_pth="output/teacher_baseline/"+config.ethnicity+'/'+config.network+'/epochs_' + str(config.num_epoch)+"/"+str(config.best_model)+"backbone.pth"
else:
    config.out_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/original/" + config.network + '/epochs_' + str(config.num_epoch)
    if config.network=="iresnet34": 
        if config.ethnicity=="Caucasian":
            config.best_model=45846
        elif config.ethnicity=="African":
            config.best_model=40512
        elif config.ethnicity=="Asian":
            config.best_model=43044
        else:
            config.best_model=34304
    elif config.network=="iresnet50":
        if config.ethnicity=="Caucasian":
            config.best_model=40752
        elif config.ethnicity=="African":
            config.best_model=50640
        elif config.ethnicity=="Asian":
            config.best_model=43044
        else:
            config.best_model=36448
    config.backbone_pth="output/"+config.ethnicity+'/'+config.network+'/epochs_'+str(config.num_epoch)+"/"+str(config.best_model)+"backbone.pth"
        
# definitions for adaptor training
if config.is_teacher_baseline:
    if config.is_dual_layer:
        if config.has_drop_out:
            config.out_arc_adapter="/nas-ctm01/homes/mecaldeira/output/EArcAdapter/baseline/dropout_dual_layer_"+str(config.middle_layer_size)+"/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
            config.out_arc_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline_embeddings/fully_connected/EArcFace/dropout_dual_layer_"+str(config.middle_layer_size)+"/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
        else:
            config.out_arc_adapter="/nas-ctm01/homes/mecaldeira/output/EArcAdapter/baseline/dual_layer_"+str(config.middle_layer_size)+"/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
            config.out_arc_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline_embeddings/fully_connected/EArcFace/dual_layer_"+str(config.middle_layer_size)+"/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method    
    else:
        config.out_arc_adapter="/nas-ctm01/homes/mecaldeira/output/EArcAdapter/baseline/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
        config.out_arc_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline_embeddings/fully_connected/EArcFace/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
else:
    if config.is_dual_layer:
        if config.has_drop_out:
            config.out_arc_adapter="/nas-ctm01/homes/mecaldeira/output/EArcAdapter/dropout_dual_layer_"+str(config.middle_layer_size)+"/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
            config.out_arc_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/fully_connected/EArcFace/dropout_dual_layer_"+str(config.middle_layer_size)+"/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
        else:
            config.out_arc_adapter="/nas-ctm01/homes/mecaldeira/output/EArcAdapter/dual_layer_"+str(config.middle_layer_size)+"/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
            config.out_arc_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/fully_connected/EArcFace/dual_layer_"+str(config.middle_layer_size)+"/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
    else:
        config.out_arc_adapter="/nas-ctm01/homes/mecaldeira/output/EArcAdapter/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method
        config.out_arc_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/fully_connected/EArcFace/"+config.network+"/lr_"+str(config.lr_fc)+"_ep_"+str(config.num_epoch_fc)+"/"+config.arc_method

# other hyperparameters
config.momentum = 0.9
config.embedding_size = 512 # embedding size of model
config.weight_decay = 5e-4
config.lr = 0.1
config.global_step=0 # step to resume
config.s=64.0
config.m=0.50
config.std=0.05

if (config.loss=="ElasticArcFacePlus"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif (config.loss=="ElasticArcFace"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
if (config.loss=="ElasticCosFacePlus"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif (config.loss=="ElasticCosFace"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05
elif (config.loss=="CurricularFace"):
    config.s = 64.0
    config.m = 0.50
elif (config.loss=="RaceFace"):
    config.s = 64.0
    config.m = 0.50