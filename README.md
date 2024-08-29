# MST-KD: Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition
Official repository for the paper **MST-KD: Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition**.

## Abstract
As in school, one teacher to cover all subjects is insufficient to distill equally robust information to a student. Hence, each subject is taught by a highly specialised teacher. Following a similar philosophy, we propose a multiple specialized teacher framework to distill knowledge to a student network. In our approach, directed at face recognition use cases, we train four teachers on one specific ethnicity, leading to four highly specialized and biased teachers. Our strategy learns a project of these four teachers into a common space and distill that information to a student network. Our results highlighted increased performance and reduced bias for all our experiments. In addition, we further show that having biased/specialized teachers is crucial by showing that our approach achieves better results than when knowledge is distilled from four teachers trained on balanced datasets. Our approach represents a step forward to the understanding of the importance of ethnicity-specific features.

## Code Usage
Here we summarize the steps replicate the work presented in the paper. The mentioned hyperparameters can be altered in the file config/config.py and **should be changed there** (some hyperparameters should be changed at specific steps to exactly mimic the paper results).

1. Run baseline_data_split.py four times (changing the cfg.ethnicity parameter) to perform a balanced datasplit of the trainset in four groups. Each subset will be used to train one of the four baseline teachers
2. Run train.py to train each of the teacher models. This script should be ran four times to obtain the four teacher models, changing the cfg.ethnicity parameter. To train the baseline teachers, set cfg.is_teacher_baseline to True
3. Run extract_all for the four ethnicities by changing the cfg.ethnicity parameter
4. To obtain the adaptors:
    - Run fc_EArc.py once to obtain SL-EAF-Fusion
    - Run dual_layer_EAF-Fusion.py twice to obtain DuL-EAF-Fusion (setting cfg.has_dropout=False) and DLDPO-EAF-Fusion (setting cfg.has_dropout=True)
5. To obtain the students, run KD_alone.py (for a-KD) or KD_CEL.py (for EAF-KD) with each of the following configurations:
    - Set cfg.is_dual_layer to False to obtain SL-EAF-Fusion students
    - Set cfg.is_dual_layer to True and cfg.has_dropout to False to obtain DuL-EAF-Fusion students
    - Set cfg.is_dual_layer to True and cfg.has_dropout to True to obtain DLDPO-EAF-Fusion students

## Training Stages

### Teachers
TODO - Image + brief description
### Adaptors
TODO - Image + brief description
### Students
TODO - Image + brief description

## Final Models
The models mentioned in the paper (teachers, adaptors and students) can be found here.

## Citation
If you use our work in your research, please cite with:

```
Soon
```
