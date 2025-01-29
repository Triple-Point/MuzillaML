MODEL_CLASSES = {
    'cosine_model': 'src.models.cosine_model.CosineModel',
    'random_model': 'src.models.random_model.RandomModel',
    'tfidf_model': 'src.models.tfidf_model.TFIDFModel',
    'popularity_model': 'src.models.popularity_model.PopularityModel'
}


def model_factory(model_type):
    model_path = MODEL_CLASSES.get(model_type)
    if not model_path:
        raise ValueError(f"Invalid model type: {model_type}")
    module_path, class_name = model_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    return model_class
