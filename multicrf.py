import argparse
import os
import random
from collections import defaultdict

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

        for epoch in range(args.epochs):
            sent = 0
            tokens = 0.0
            cum_loss = 0
            correct = 0
            print("Starting epoch %d .." % epoch)
            batches = utils.make_bucket_batches(
                zip(sents, char_sents, langs, sent_index), tgt_tags, args.batch_size
            )
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
            print("Evaluating on dev set...")

    else:
        print("Loading tagger model from " + args.model_name + "...")
        tagger_model = torch.load(
            args.model_name, map_location=lambda storage, loc: storage
        )
        if args.gpu:
            tagger_model = tagger_model.cuda()

    if args.test:
        avg_tok_accuracy, f1_score = eval(tagger_model, dev_or_test="test")


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
    parser.add_argument("--patience", type=int, default=3)
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
    main()
