from texttable import Texttable
import torch


def viz_turns(turns, show=True):
    t = Texttable(max_width=150)
    t.add_rows([["turn", "context"]])
    for idx, data in enumerate(turns):
        text = "id: {}\n".format(data.id)
        text += "transcript: {}\n".format(data.transcript)
        text += "turn_label: {}\n".format(data.turn_label)
        text += "belief_state: {}\n".format(data.belief_state)
        text += "system_acts: {}\n".format(data.system_acts)
        text += "system_transcript: {}\n".format(data.system_transcript)
        t.add_row([str(idx), text])
    if show:
        print(t.draw())
    return t.draw()


def get_turn_pred_gold(gold_request, gold_inform, gold_recovered, pred_request, pred_inform,
                       pred_recovered):
    # print("gold_request: {}".format(gold_request))
    # print("gold_inform: {}".format(gold_inform))
    # print("gold_recovered: {}".format(gold_recovered))
    gold = "gold_request: {}\n".format(gold_request)
    gold += "gold_inform: {}\n".format(gold_inform)
    gold += "gold_recovered: {}\n".format(gold_recovered)

    pred = "pred_request: {}\n".format(pred_request)
    pred += "pred_inform: {}\n".format(pred_inform)
    pred += "pred_recovered: {}\n".format(pred_recovered)
    return gold, pred


def viz_embddings(batch, emb):
    tee = Texttable(max_width=150)
    tee.add_rows([["Attribute", "content"]])

    seqs = [e.num['transcript'] for e in batch]
    utterance_len = [len(s) for s in seqs]
    max_len = max(utterance_len)
    utterance_padded = torch.LongTensor(
        [s + (max_len - l) * [1] for s, l in zip(seqs, utterance_len)])
    utterance = emb(utterance_padded.to('cpu'))

    # system_acts_len = [len(s) for s in seqs_system_acts]
    # max_len = max(system_acts_len)
    # system_acts_padded = torch.LongTensor([s + (max_len - l) * [1] for s, l in zip(seqs_system_acts, system_acts_len)])
    # system_acts = emb(system_acts_padded.to('cpu'))

    trans = ''
    acts_text = ''
    utterance_0 = str(utterance[0]) + "\n" + str(utterance.size()) + "\n" + str(utterance_len)
    # system_acts_0 = str(system_acts[0]) + "\n" + str(system_acts.size()) + "\n" + str(system_acts_len)
    for b in batch:
        trans = trans + str(b.transcript) + "\n" + str(b.num['transcript']) + "\n"
        acts_text = acts_text + str(b.system_acts) + "\n" + str(b.num['system_acts']) + "\n"
    tee.add_row(["transcript", trans])
    tee.add_row(["system_acts", acts_text])
    tee.add_row(["utterance_padded", utterance_padded])
    tee.add_row(["utterance", utterance_0])

    system_acts_padded = ''
    system_acts = []
    # seqs_system_acts = [e.num['system_acts'] for e in batch]
    for e in batch:
        seqs = e.num['system_acts']
        seqs_len = [len(s) for s in seqs]
        max_len = max(seqs_len)
        padded = torch.LongTensor([s + (max_len - l) * [1] for s, l in zip(seqs, seqs_len)])

        embs = emb(padded.to('cpu'))

        system_acts_padded = system_acts_padded + str(padded) + "\n"
        system_acts.append(embs)
    system_acts_0 = [e.size() for e in system_acts]
    tee.add_row(["system_acts_padded", system_acts_padded])
    tee.add_row(["system_acts", system_acts_0])
    print(tee.draw())


def viz_inputs_labels(ontology, batch):
    tee = Texttable(max_width=150)
    tee.add_rows([["Attribute", "content"]])
    inputs = "transcript: \n" + str([e.transcript for e in batch]) + "\n"
    inputs = inputs + "system_acts:\n " + str([e.system_acts for e in batch])
    labels = {s: [len(ontology.values[s]) * [0] for i in range(len(batch))] for s in ontology.slots}
    tee.add_row(["labels", labels])

    print(tee.draw())


if __name__ == "__main__":
    # reformat_json("/Users/hao/Projects/Ftech/glad/data/woz/ann/dev.json")
    # vocab = load_json("/Users/hao/Projects/Ftech/glad/data/woz/ann/vocab.json")
    # words = vocab["index2word"]
    # i2w = {i: w for i, w in enumerate(words)}
    # write_json_beutifier("/Users/hao/Projects/Ftech/glad/data/woz/ann/idx2word.json", i2w)
    preds_all = [{('request', 'address'),
                  ('area', 'center'),
                  ('food', 'spanish'),
                  ('request', 'phone')},
                 set(),
                 set(),
                 set(),
                 set(),
                 {('food', 'korean')},
                 {('request', 'address'), ('request', 'phone')},
                 set()]

    for preds in [preds_all[0: 5], preds_all[5:]]:
        # preds = preds[t:]
        pred_state = {}
        for i in range(len(preds)):
            pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])

            pred_recovered = set()
            for s, v in pred_inform:
                pred_state[s] = v
            for s, v in pred_state.items():
                pred_recovered.add(('inform', s, v))
            print(pred_inform)
            # print(pred_recovered)
