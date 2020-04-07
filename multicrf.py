import argparse
import codecs
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim

import dataloader
import models
import utils


def main():

    if not os.path.isfile(args.model_name) or args.continue_train:
        if args.continue_train:
            print("Loading tagger model from " + args.model_name + "...")
            tagger_model = torch.load(
                args.model_name, map_location=lambda storage, loc: storage
            )

        else:
            tagger_model = models.BiLSTMCRFTagger

            tagger_model = tagger_model(
                args.emb_dim,
                args.hidden_dim,
                len(data_loader.char_to_id),
                len(data_loader.word_to_id),
                tagset_sizes=data_loader.tag_vocab_sizes,
                n_layers=args.n_layers,
                dropOut=args.dropout,
                gpu=args.gpu,
            )
        if args.gpu:
            tagger_model = tagger_model.cuda()

        if args.optim == "sgd":
            optimizer = optim.SGD(tagger_model.parameters(), lr=0.1)
        elif args.optim == "adam":
            optimizer = optim.Adam(tagger_model.parameters())
        elif args.optim == "adagrad":
            optimizer = optim.Adagrad(tagger_model.parameters())
        elif args.optim == "rmsprop":
            optimizer = optim.RMSprop(tagger_model.parameters())

        print("Training tagger model...")
        prev_avg_tok_accuracy = 0
        patience_counter = 0

        for epoch in range(args.epochs):
            sent = 0
            tokens = 0.0
            cum_loss = 0
            correct = 0
            print("Starting epoch %d .." % epoch)
            batches = utils.make_bucket_batches(
                zip(sents, char_sents, langs, sent_index), tgt_tags, args.batch_size
            )
            tagger_model.train()
            for _, b_char_sents, _, _, b_tgt_tags in batches:
                tagger_model.zero_grad()
                b_char_sents_tensor = []
                for character_sentence in b_char_sents:
                    b_char_sents_tensor.append([])
                    for word in character_sentence:
                        word = torch.LongTensor(word)
                        if args.gpu:
                            word = word.cuda()
                        b_char_sents_tensor[-1].append(word)
                b_tags = {
                    feature: torch.LongTensor(
                        [b_tgt_tags[feature][i] for i in range(len(b_char_sents))]
                    )
                    for feature in data_loader.tag_vocab_sizes
                }

                # get prediction
                tag_dict = tagger_model(b_char_sents_tensor)

                # accuracy
                correct += sum(
                    [
                        (out_tags["states"] == b_tags[tag]).sum().item()
                        for tag, out_tags in tag_dict.items()
                    ]
                )
                tokens += (
                    len(b_char_sents)
                    * tag_dict["POS"]["states"].shape[1]
                    * len(tag_dict)
                )

                # loss
                loss = tagger_model.neg_log_likelihood(tag_dict, b_tags)
                cum_loss += loss.detach().cpu().item()
                loss.backward()
                optimizer.step()

                if sent % 100 == 0:
                    print(
                        "[Epoch %d] \
                        Batches %d/%d, \
                        Tokens %d \
                        Cum_Loss: %f \
                        Average Accuracy: %f"
                        % (
                            epoch,
                            sent,
                            len(batches),
                            tokens,
                            cum_loss / tokens,
                            correct / tokens,
                        )
                    )

                sent += 1

            print("Loss: %f" % loss.detach().cpu().numpy())
            print("Accuracy: %f" % (correct / tokens))
            print("Saving model..")
            torch.save(tagger_model, args.model_name)

            if epoch < 2:
                # skip evaluation the first two epochs
                continue
            print("Evaluating on dev set...")
            with torch.no_grad():
                tagger_model.eval()
                avg_tok_accuracy = eval(
                    tagger_model, curEpoch=epoch, dev_or_test="dev"
                )[0]

            # Early Stopping
            if avg_tok_accuracy <= prev_avg_tok_accuracy:
                patience_counter += 1
                if patience_counter == args.patience:
                    print(
                        "Model hasn't improved on dev set for %d epochs. Stopping Training."
                        % patience_counter
                    )
                    break
            else:
                patience_counter = 0
            prev_avg_tok_accuracy = max(prev_avg_tok_accuracy, avg_tok_accuracy)

    else:
        print("Loading tagger model from " + args.model_name + "...")
        tagger_model = torch.load(
            args.model_name, map_location=lambda storage, loc: storage
        )
        if args.gpu:
            tagger_model = tagger_model.cuda()
        eval(tagger_model)

    if args.test:
        tagger_model.eval()
        avg_tok_accuracy, f1_score = eval(tagger_model, dev_or_test="test")


def eval(tagger_model, curEpoch=None, dev_or_test="dev"):
    eval_data = (
        (dev_sents, dev_char_sents, dev_tgt_tags)
        if dev_or_test == "dev"
        else (test_sents, test_char_sents, test_tgt_tags)
    )
    correct = 0
    print(
        "Starting evaluation on %s set... (%d sentences)"
        % (dev_or_test, len(eval_data[0]))
    )
    (sents, char_sents, tgt_tags) = eval_data
    prefix = args.model_type + "_"
    if dev_or_test == "dev":
        prefix += (
            "-".join([l for l in all_langs]) + "_" + dev_or_test + "_" + str(curEpoch)
        )
    else:
        prefix += "-".join([l for l in all_langs]) + "_" + dev_or_test
    predictions, sentences, tokens, goldTags = [], [], 0, []
    with codecs.open(prefix, "w", encoding="utf-8") as fout:

        batches = utils.make_bucket_batches(
            zip(sents, char_sents, langs, sent_index), tgt_tags, 1, shuffle=False
        )
        for sent, char_sent, _, _, tgt_tag in batches:
            tagger_model.zero_grad()
            b_char_sents_tensor = []
            for character_sentence in char_sent:
                b_char_sents_tensor.append([])
                for word in character_sentence:
                    word = torch.LongTensor(word)
                    if args.gpu:
                        word = word.cuda()
                    b_char_sents_tensor[-1].append(word)
            b_tags = {
                feature: torch.LongTensor(
                    [tgt_tag[feature][i] for i in range(len(char_sent))]
                )
                for feature in data_loader.tag_vocab_sizes
            }

            # get prediction
            tag_dict = tagger_model(b_char_sents_tensor)
            one_gold = [{} for _ in range(len(char_sent[0]))]
            one_prediction = [{} for _ in range(len(char_sent[0]))]

            for tag, out_tags in tag_dict.items():
                best_path = out_tags["states"].data.cpu().numpy()[0]
                assert len(best_path) == len(tgt_tag[tag][0])
                for token_num, pred_tag_feat in enumerate(best_path):
                    if data_loader.id2tags[tag][pred_tag_feat] == "_":
                        one_prediction[token_num][tag] = "NULL"
                        continue
                    tag_name = data_loader.id2tags[tag][pred_tag_feat]
                    one_prediction[token_num][tag] = tag_name

                for token_num, gold_tag_feat in enumerate(tgt_tag[tag][0]):
                    if data_loader.id2tags[tag][gold_tag_feat] == "_":
                        one_gold[token_num][tag] = "NULL"
                        continue
                    tag_name = data_loader.id2tags[tag][gold_tag_feat]
                    one_gold[token_num][tag] = tag_name

            for word_dict in one_prediction:
                predictions.append(word_dict)
            for word_dict in one_gold:
                goldTags.append(word_dict)
            sentences.append([data_loader.id_to_word[id] for id in sent[0]])

            for token_num in range(1, len(sent[0]) - 1):
                info = ["_" for _ in range(10)]
                info[0] = str(token_num)
                info[1] = data_loader.id_to_word[sent[0][token_num]]
                pred_tagdict = one_prediction[token_num]
                gold_tagdict = one_gold[token_num]
                pred_tags = []
                gold_tags = []
                for k, v in pred_tagdict.items():
                    if v != "NULL":
                        pred_tags.append(k + "=" + v)
                for k, v in gold_tagdict.items():
                    if v != "NULL":
                        gold_tags.append(k + "=" + v)

                if len(pred_tags) == 0:
                    info[5] = "_"
                else:
                    info[5] = "|".join(pred_tags)

                if len(gold_tags) == 0:
                    info[6] = "_"
                else:
                    info[6] = "|".join(gold_tags)
                fout.write("\t".join(info) + "\n")
            fout.write("\n")

            # accuracy
            correct += sum(
                [
                    (out_tags["states"] == b_tags[tag]).sum().item()
                    for tag, out_tags in tag_dict.items()
                ]
            )
            tokens += (
                len(char_sent) * tag_dict["POS"]["states"].shape[1] * len(tag_dict)
            )

    avg_tok_accuracy = correct / tokens
    f1_score, f1_micro_score = utils.computeF1(
        predictions, goldTags, prefix, write_results=True
    )
    print(f"{dev_or_test} Set Accuracy: {avg_tok_accuracy}")
    print(f"{dev_or_test} Set Avg F1 Score (Macro): {f1_score}")
    print(f"{dev_or_test} Set Avg F1 Score (Micro): {f1_micro_score}")

    (acc, f1, pre, recall) = utils.manipulate_data(goldTags, predictions)
    print(f"New {dev_or_test} Set Accuracy: {acc}")
    print(f"New {dev_or_test} Set F1 Score: {f1}")
    print(f"New {dev_or_test} Set P: {pre}")
    print(f"New {dev_or_test} Set R: {recall}")
    return acc, f1
    # return avg_tok_accuracy, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--treebank_path",
        type=str,
        default="/Users/aditichaudhary/Documents/CMU/Typ_dara/",
    )
    parser.add_argument(
        "--optim", type=str, default="adam", choices=["sgd", "adam", "adagrad"]
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--langs",
        type=str,
        default="mr",
        help="Languages separated by delimiter '/' with last language being target language",
    )
    parser.add_argument(
        "--tgt_size",
        type=int,
        default=None,
        help="Number of training sentences for target language",
    )
    parser.add_argument("--model_name", type=str, default="model_dcrf")
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=["universal", "joint", "mono", "specific", "baseline"],
    )
    parser.add_argument("--sum_word_char", action="store_true")
    parser.add_argument("--sent_attn", action="store_true")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--unit_test", action="store_true")
    parser.add_argument("--unit_test_args", type=str, default="2,2,2")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print(args)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    all_langs = args.langs.split("/")
    args.model_name += (
        "_" + args.model_type + "".join(["_" + l for l in all_langs])
    )  # Set model name
    if args.sent_attn:
        args.model_name += "-sent_attn"
    if args.tgt_size:
        args.model_name += "-" + str(args.tgt_size)

    data_loader = dataloader.DataLoader(args)

    sents, char_sents, tgt_tags, langs = [], [], defaultdict(list), []
    for lang in all_langs:
        input_folder = (
            args.treebank_path + "/" + "UD_" + data_loader.code_to_lang[lang] + "//"
        )
        print("Reading files from folder", input_folder)
        for [path, dir, files] in os.walk(input_folder):
            for file in files:
                if "train" in file and file.endswith(".conllu"):
                    train_path = input_folder + file
                    print("Reading from, ", train_path)
                    break

        lang_sents, lang_char_sents, lang_tgt_tags = data_loader.get_data_set(
            train_path, lang
        )
        # Oversample target language data
        if args.tgt_size == 100 and args.model_type != "mono" and lang == all_langs[-1]:
            sents += lang_sents * 10
            char_sents += lang_char_sents * 10
            for key, lang_tags in lang_tgt_tags.items():
                tgt_tags[key] += lang_tags * 10
            langs += ["<" + lang + ">" for _ in range(len(lang_sents) * 10)]
        else:
            sents += lang_sents
            char_sents += lang_char_sents
            for key, lang_tags in lang_tgt_tags.items():
                tgt_tags[key] += lang_tags
            langs += ["<" + lang + ">" for _ in range(len(lang_sents))]

    sent_index = [i for i in range(len(sents))]
    print("Data set size (train): %d" % len(sents))

    lang = all_langs[-1]
    for file in os.listdir(args.treebank_path + "UD_" + data_loader.code_to_lang[lang]):
        if file.endswith("dev.conllu"):
            dev_path = os.path.join(
                args.treebank_path + "UD_" + data_loader.code_to_lang[lang], file
            )

        elif file.endswith("test.conllu"):
            test_path = os.path.join(
                args.treebank_path + "UD_" + data_loader.code_to_lang[lang], file
            )

    dev_sents, dev_char_sents, dev_tgt_tags = data_loader.get_data_set(
        dev_path, all_langs[-1]
    )
    test_sents, test_char_sents, test_tgt_tags = data_loader.get_data_set(
        test_path, all_langs[-1]
    )
    feature = "POS"

    main()
