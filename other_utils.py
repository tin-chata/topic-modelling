"""
Created on 2019-01-14
@author: duytinvo
"""
import time
import gzip
import sys
import pickle
import math
import numpy as np

# source: https://www.ranks.nl/stopwords
sws = "a able about above abst accordance according accordingly across act actually added adj affected affecting " \
      "affects after afterwards again against ah all almost alone along already also although always am among " \
      "amongst an and announce another any anybody anyhow anymore anyone anything anyway anyways anywhere " \
      "apparently approximately are aren arent arise around as aside ask asking at auth available away awfully b " \
      "back be became because become becomes becoming been before beforehand begin beginning beginnings begins " \
      "behind being believe below beside besides between beyond biol both brief briefly but by c ca came can cannot " \
      "can't cause causes certain certainly co com come comes contain containing contains could couldnt d date did " \
      "didn't different do does doesn't doing done don't down downwards due during e each ed edu effect eg eight " \
      "eighty either else elsewhere end ending enough especially et et-al etc even ever every everybody everyone " \
      "everything everywhere ex except f far few ff fifth first five fix followed following follows for former " \
      "formerly forth found four from further furthermore g gave get gets getting give given gives giving go goes " \
      "gone got gotten h had happens hardly has hasn't have haven't having he hed hence her here hereafter hereby " \
      "herein heres hereupon hers herself hes hi hid him himself his hither home how howbeit however hundred i id ie " \
      "if i'll im immediate immediately importance important in inc indeed index information instead into invention " \
      "inward is isn't it itd it'll its itself i've j just k keep keeps kept kg km know known knows l largely last " \
      "lately later latter latterly least less lest let lets like liked likely line little 'll look looking looks " \
      "ltd m made mainly make makes many may maybe me mean means meantime meanwhile merely mg might million miss " \
      "ml more moreover most mostly mr mrs much mug must my myself n na name namely nay nd near nearly necessarily " \
      "necessary need needs neither never nevertheless new next nine ninety no nobody non none nonetheless noone " \
      "nor normally nos not noted nothing now nowhere o obtain obtained obviously of off often oh ok okay old " \
      "omitted on once one ones only onto or ord other others otherwise ought our ours ourselves out outside " \
      "over overall owing own p page pages part particular particularly past per perhaps placed please plus " \
      "poorly possible possibly potentially pp predominantly present previously primarily probably promptly proud " \
      "provides put q que quickly quite qv r ran rather rd re readily really recent recently ref refs regarding " \
      "regardless regards related relatively research respectively resulted resulting results right run s said " \
      "same saw say saying says sec section see seeing seem seemed seeming seems seen self selves sent seven " \
      "several shall she shed she'll shes should shouldn't show showed shown showns shows significant " \
      "significantly similar similarly since six slightly so some somebody somehow someone somethan something " \
      "sometime sometimes somewhat somewhere soon sorry specifically specified specify specifying still stop " \
      "strongly sub substantially successfully such sufficiently suggest sup sure t take taken taking tell tends " \
      "th than thank thanks thanx that that'll thats that've the their theirs them themselves then thence there " \
      "thereafter thereby thered therefore therein there'll thereof therere theres thereto thereupon there've " \
      "these they theyd they'll theyre they've think this those thou though thoughh thousand throug through " \
      "throughout thru thus til tip to together too took toward towards tried tries truly try trying ts twice " \
      "two u un under unfortunately unless unlike unlikely until unto up upon ups us use used useful usefully " \
      "usefulness uses using usually v value various 've very via viz vol vols vs w want wants was wasnt way we " \
      "wed welcome we'll went were werent we've what whatever what'll whats when whence whenever where whereafter " \
      "whereas whereby wherein wheres whereupon wherever whether which while whim whither who whod whoever " \
      "whole who'll whom whomever whos whose why widely willing wish with within without wont words world " \
      "would wouldnt www x y yes yet you youd you'll your youre yours yourself yourselves you've z zero"
sw_set = set(sws.split())


# --------------------------------------------------------------------------------------------------------------------
# ======================================== UTILITY FUNCTIONS =========================================================
# --------------------------------------------------------------------------------------------------------------------
class Timer:
    @staticmethod
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    @staticmethod
    def asHours(s):
        h = math.floor(s / 3600)
        m = math.floor((s - h * 3600) / 60)
        s -= (h * 3600 + m * 60)
        return '%dh %dm %ds' % (h, m, s)

    @staticmethod
    def timeSince(since):
        now = time.time()
        s = now - since
        return '%s' % (Timer.asMinutes(s))

    @staticmethod
    def timeEst(since, percent):
        s = time.time() - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (Timer.asMinutes(s), Timer.asHours(rs))


# Save and load hyper-parameters
class SaveloadHP:
    @staticmethod
    def save(args, argfile='./results/model_args.pklz'):
        """
        argfile='model_args.pklz'
        """
        print("Writing hyper-parameters into %s" % argfile)
        with gzip.open(argfile, "wb") as fout:
            pickle.dump(args, fout, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(argfile='./results/model_args.pklz'):
        print("Reading hyper-parameters from %s" % argfile)
        with gzip.open(argfile, "rb") as fin:
            args = pickle.load(fin)
        return args


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)

