import torch
import torch.nn as nn
from torch.autograd import Variable

from PIL import Image
import example_pb2
from meta import Meta
import lmdb


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

    
# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


'''Unwrapped Rewards'''
def convert_to_lmdb(img, rwd, lmdb_path, idx):
    img = Image.fromarray(img, 'RGB')
    img = img.crop((0, 8, 67, 22))  # SpaceInvadersNoFrameskip-v4
    img.save(lmdb_path + '/test/' + str(idx) + '.png')
    img = img.resize([64, 64]).tobytes()

    # reward to digits
    label_of_digits = list(str(int(rwd)))
    length = len(label_of_digits)
    digits = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]  # digit 10 represents no digit
    for idx, label_of_digit in enumerate(label_of_digits):
        digits[idx] = int(label_of_digit if label_of_digit != 10 else 0)  # label 10 is essentially digit zero

    example = example_pb2.Example()
    example.image = img
    example.length = length
    example.digits.extend(digits)

    str_id = '{:08}'.format(idx)

    env = lmdb.open(lmdb_path + '/test.lmdb', map_size=10*1024*1024*1024)
    with env.begin(write=True) as txn:
        txn.put(str_id.encode(), example.SerializeToString())


def create_lmdb_meta_file(num_test_examples, path_to_lmdb_meta_file):
    print('Saving meta file to %s...' % path_to_lmdb_meta_file)
    meta = Meta()
    # meta.num_train_examples = num_train_examples
    # meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_lmdb_meta_file)


def digit_eval(image, length_labels, digits_labels, model):
    model.eval()
    num_correct = 0
    needs_include_length = False

    for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
        # images, length_labels, digits_labels = (Variable(images.cuda(), volatile=True),
        #                                         Variable(length_labels.cuda()),
        #                                         [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
        length_labels, digits_labels = (Variable(length_labels.cuda()),
                                        [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
        with torch.no_grad():
            images = Variable(images.cuda())
        length_logits, digits_logits = model(images)
        length_predictions = length_logits.data.max(1)[1]
        digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

        if needs_include_length:
            num_correct += (length_predictions.eq(length_labels.data) &
                            digits_predictions[0].eq(digits_labels[0].data) &
                            digits_predictions[1].eq(digits_labels[1].data) &
                            digits_predictions[2].eq(digits_labels[2].data) &
                            digits_predictions[3].eq(digits_labels[3].data) &
                            digits_predictions[4].eq(digits_labels[4].data) &
                            digits_predictions[5].eq(digits_labels[5].data) &
                            digits_predictions[6].eq(digits_labels[6].data) &
                            digits_predictions[7].eq(digits_labels[7].data) &
                            digits_predictions[8].eq(digits_labels[8].data) &
                            digits_predictions[9].eq(digits_labels[9].data)).cpu().sum()
        else:
            num_correct += (digits_predictions[0].eq(digits_labels[0].data) &
                            digits_predictions[1].eq(digits_labels[1].data) &
                            digits_predictions[2].eq(digits_labels[2].data) &
                            digits_predictions[3].eq(digits_labels[3].data) &
                            digits_predictions[4].eq(digits_labels[4].data) &
                            digits_predictions[5].eq(digits_labels[5].data) &
                            digits_predictions[6].eq(digits_labels[6].data) &
                            digits_predictions[7].eq(digits_labels[7].data) &
                            digits_predictions[8].eq(digits_labels[8].data) &
                            digits_predictions[9].eq(digits_labels[9].data)).cpu().sum()

    accuracy = num_correct.item() / float(len(self._loader.dataset))
    return accuracy
