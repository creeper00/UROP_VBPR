import cornac
from cornac.datasets import amazon_clothing
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit
from vbpr import VBPR


feedback = amazon_clothing.load_feedback()
features, item_ids = amazon_clothing.load_visual_feature()  # BIG file
item_image_modality = ImageModality(features=features, ids=item_ids, normalized=True)
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
    item_image=item_image_modality,
)

# VBPR
vbpr = VBPR(
    k=10,
    k2=20,
    n_epochs=50,
    batch_size=100,
    learning_rate=0.005,
    lambda_w=1,
    lambda_b=0.01,
    lambda_e=0.0,
    use_gpu=True,
)

auc = cornac.metrics.AUC()
rec_50 = cornac.metrics.Recall(k=50)

# Evaluation
cornac.Experiment(eval_method=ratio_split, models=[vbpr], metrics=[auc, rec_50]).run()
