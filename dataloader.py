from collections import defaultdict
import utils, os, codecs

class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.lang_to_code, self.code_to_lang = utils.get_lang_code_dicts()
        paths_to_read = []
        langs = args.langs.split("/")
        for lang in langs:
            input_folder = args.treebank_path + "/" + "UD_" + self.code_to_lang[lang] + "//"
            for [path, dir, files] in os.walk(input_folder):

                files.sort()
                for file in files:
                    if file.endswith(".conllu"):
                        path = input_folder + file
                        print("Reading vocab from ", path)
                        paths_to_read.append((path, lang))
                break

        self.tag_to_ids, self.word_to_id, self.char_to_id= self.read_files(paths_to_read)
        print("Size of vocab before: %d" % len(self.word_to_id))
        self.word_to_id['<unk>'] = len(self.word_to_id)
        self.char_to_id['<unk>'] = len(self.char_to_id)

        self.word_to_id['<\s>'] = len(self.word_to_id)
        self.char_to_id['<pad>'] = len(self.char_to_id)
        print("Size of vocab after: %d" % len(self.word_to_id))
        self.word_padding_token = 0
        self.char_padding_token = 0
        self.id2tags = {}
        self.tag_vocab_sizes = {}
        self.word_freq = {}
        for key, tag2id in self.tag_to_ids.items():
            self.id2tags[key] = {v: k for k, v in tag2id.items()}
            self.tag_vocab_sizes[key] = len(tag2id)
            print("Feat: {0} Size: {1}".format(key, len(tag2id)))
            print(self.tag_to_ids[key])
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        self.word_vocab_size = len(self.id_to_word)
        self.char_vocab_size = len(self.id_to_char)
        print("Size of vocab after: %d" % len(self.word_to_id))
        print("Word vocab size=%d, Char Vocab size=%d" % (self.word_vocab_size, self.char_vocab_size))

    def get_vocab_from_set(self, a_set, shift=0):
        vocab = {}
        for i, elem in enumerate(a_set):
            vocab[elem] = i + shift

        return vocab

    def get_vocab_from_dict(self, a_dict):
        i=0
        vocab = {}
        for k,v in a_dict.items():
            vocab[k] = i
            i += 1
        return vocab

    def read_files(self, paths):
        # word_list = []
        # char_list = []
        # tag_list = []
        word_dict = defaultdict(lambda: 0)
        char_set = set()
        tag_vocab = defaultdict(set)

        def _read_a_file(path, lang):
            fdebug = codecs.open("./debug.txt", "w", encoding='utf-8')
            with codecs.open(path, "r", "utf-8") as fin:
                to_read_line = []
                line = ["_" for _ in range(10)]
                line[1] = "<" + lang + ">"
                to_read_line.append("\t".join(line))
                for line in fin:
                    if line.startswith("#"):
                        continue
                    if line.strip() == "":
                        line = ["_" for _ in range(10)]
                        line[1] = "<" + lang + ">"
                        to_read_line.append("\t".join(line))
                        self.read_one_line(to_read_line, tag_vocab, word_dict, char_set, fdebug, lang)
                        to_read_line = []
                        line = ["_" for _ in range(10)]
                        line[1] = "<" + lang + ">"
                        to_read_line.append("\t".join(line))
                    else:
                        to_read_line.append(line.strip())
                self.read_one_line(to_read_line, tag_vocab, word_dict, char_set, fdebug, lang)

        for (path, lang) in paths:
            _read_a_file(path, lang)

        tag2ids = {}
        for key, tag_set in tag_vocab.items():
            if key == "_":
                continue
            tag_set.add("_")
            tag2ids[key] = self.get_vocab_from_set(tag_set)
        word_vocab = self.get_vocab_from_dict(word_dict)
        char_vocab = self.get_vocab_from_set(char_set, 0)

        return tag2ids, word_vocab, char_vocab


    def read_one_line(self, line, tag_set, word_dict, char_set, fdebug, lang):
        write = False
        for w in line:
            fields = w.split("\t")
            if len(fields) != 10:
                print("ERROR")
                print(fields)
                exit(0)
            word = fields[1]
            feats = fields[5]
            pos = fields[3]
            if feats == "":
                print("ERROR: the feature is blank, re-run pretrain_pos")
                write=True
            for c in word:
                char_set.add(c)
            tag_set['POS'].add(pos)
            for feat in feats.split("|"):
                feat_info = feat.split("=")
                tag_set[feat_info[0]].add(feat_info[-1])
            word_dict[word] += 1
        if write:
            for w in line:
                fdebug.write(w + "\n")
            fdebug.write("\n")
            exit(-1)

    def get_data_set(self, path, lang):
        sents = []
        char_sents = []
        tgt_tags = defaultdict(list)
        discrete_features = []
        bc_features = []
        known_tags = defaultdict(list)

        def add_sent(one_sent):
            temp_sent = []
            temp_feats = defaultdict(list)
            temp_char = []
            sent = []

            line = ["_" for _ in range(10)]
            line[1] = "<"  + lang  + ">"
            one_sent  = ["\t".join(line)] + one_sent + ["\t".join(line)] # Adding language tag before and after each sequence
            for w in one_sent:
                fields = w.split("\t")
                assert len(fields) == 10
                word = fields[1]

                if word not in self.word_freq:
                    self.word_freq[self.word_to_id[word]] = 1
                else:
                    self.word_freq[self.word_to_id[word]] += 1

                sent.append(word)
                feats = fields[5]
                pos = fields[3]
                if feats == "_": #No Morph tags, for each feature assingn "_"
                    for key in self.tag_to_ids.keys():
                        temp_feats[key].append(self.tag_to_ids[key]["_"])
                else:
                    addedFeats = set()
                    keyvalue_set = list(set(feats.split("|")))
                    temp_feats["POS"].append(self.tag_to_ids["POS"][pos])
                    addedFeats.add("POS")
                    for feat in keyvalue_set:
                        key_info = feat.split("=")
                        key = key_info[0]
                        if key in addedFeats:
                            continue

                        temp_feats[key].append(self.tag_to_ids[key][key_info[-1]])
                        addedFeats.add(key)
                    for key in self.tag_to_ids.keys():

                        if key not in addedFeats:#Adding Null char for features absent in one token
                            temp_feats[key].append(self.tag_to_ids[key]["_"])

                if word in self.word_to_id:
                    temp_sent.append(self.word_to_id[word])
                elif word.lower() in self.word_to_id:
                    temp_sent.append(self.word_to_id[word.lower()])
                else:
                    temp_sent.append(self.word_to_id["<unk>"])


                temp_char.append(
                    [self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])

            sents.append(temp_sent)
            char_sents.append(temp_char)
            for key, one_sent_feature_wise_tags in temp_feats.items():
                tgt_tags[key].append(one_sent_feature_wise_tags)


        with codecs.open(path, "r", "utf-8") as fin:
            i = 0
            one_sent = []
            for line in fin:
                if line.startswith("#"):
                    continue
                if line.strip() == "":
                    if len(one_sent) > 0:
                        add_sent(one_sent)
                        i += 1
                        if i % 1000 == 0:
                            print("Processed %d training data." % (i,))
                    one_sent = []
                else:
                    one_sent.append(line.strip())

            if len(one_sent) > 0:
                add_sent(one_sent)
                i += 1

        return sents, char_sents, tgt_tags