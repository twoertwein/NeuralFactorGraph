import argparse, codecs


def manipulate_data(golds, hyps):
#    log.info("Lemma acc, Lemma Levenshtein, morph acc, morph F1")

    count = 0
    morph_acc = 0

    f1_precision_scores = 0
    f1_precision_counts = 0
    f1_recall_scores = 0
    f1_recall_counts = 0

    for r, o in zip(golds, hyps):
        #log.debug("{}\t{}\t{}\t{}".format(r.LEMMA, o.LEMMA, r.FEATS, o.FEATS))
        gold_ = set()
        hyp_ = set()
        for k,v in r.items():
            if v == "NULL":
                continue
            gold_.add(v)
        for k,v in o.items():
            if v == "NULL":
                continue
            hyp_.add(v)
        count += 1
        morph_acc += gold_ == hyp_
        union_size = len(gold_ & hyp_)
        reference_size = len(gold_)
        output_size = len(hyp_)

        f1_precision_scores += union_size
        f1_recall_scores += union_size
        f1_precision_counts += output_size
        f1_recall_counts += reference_size

    f1_precision = f1_precision_scores / (f1_precision_counts or 1)
    f1_recall = f1_recall_scores / (f1_recall_counts or 1)
    f1 = 2 * (f1_precision * f1_recall) / (f1_precision + f1_recall + 1E-20)

    return (100 * morph_acc / count, 100 * f1, 100 * f1_precision, 100 * f1_recall)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    args = parser.parse_args()
    golds, hyps = [],[]

    with codecs.open(args.input, "r", encoding='utf-8') as fin:
        for line in fin:
            if line == "" or line == "\n":
                continue
            else:
                info = line.strip().split("\t")
                gold_data = info[6]
                pred_data = info[5]

                p,g = {},{}
                for feat in pred_data.split("|"):
                    key, value = feat.split("=")[0], feat.split("=")[1]
                    if value == "_":
                        value = "NULL"
                    p[key] = value
                for feat in gold_data.split("|"):
                    key, value = feat.split("=")[0], feat.split("=")[1]
                    if value == "_":
                        value = "NULL"
                    g[key] = value

                golds.append(g)
                hyps.append(p)

    (acc, f1, p, r) = manipulate_data(golds, hyps)
    print(acc, f1)



