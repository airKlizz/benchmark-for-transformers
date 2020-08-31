def get_value(score, value):
    if value == "rouge1":
        return score["rouge1"][1][2]
    if value == "rouge2":
        return score["rouge2"][1][2]
    if value == "rougeL":
        return score["rougeL"][1][2]
    if value == "accuracy":
        try:
            return score["accuracy"]
        except:
            return score["overall_accuracy"]
    if value == "precision":
        return score["overall_precision"]
    if value == "recall":
        return score["overall_recall"]
    if value == "f1":
        return score["overall_f1"]
    if value == "kendalltau":
        return score["tau"]
    if value == "pmr":
        return score["pmr"]
    raise ValueError(f"Metric {value} not defined.")
