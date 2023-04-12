import random
import unittest

import numpy as np

class Random_generator(object):

    def __init__(self, seed=None):
        """
        Initializes Random_generator module

        :param seed: The seed value of the random generator
        """

        self.generators = {}
        self.masterseed = np.random

        if seed != None:
            self.masterseed.seed(seed)
        else:
            self.masterseed.seed(int(str(np.random.random())[2:8]))

        self.psn=[]


    def remove_generator(self, title):
        self.generators.pop(title)

    def generate_seed(self):
        """
        Generates random seed number from master seed

        :return: Random seed number
        """

        return self.masterseed.randint(0, 999999999)


    def add_generator_int(self, title, start= 0, end = 100):
        """
        Add a random integer generator

        :param title: name of generator
        :param start: lower limit for random integer
        :param end: upper limit for random integer
        """

        if isinstance(title,str):
            rand = random.Random()
            seed = self.generate_seed()
            rand.seed(seed)
            self.generators[title] = {
                'generator' :rand,
                'type' :'int',
                'start' :  start,
                'end' :end }

    def add_generator_percentage(self, title, percentage):
        """
        Add a random integer generator

        :param title: name of generator
        :param start: lower limit for random integer
        :param end: upper limit for random integer
        """

        if isinstance(title,str):
            rand = random.Random()
            seed = self.generate_seed()
            rand.seed(seed)
            self.generators[title] = {
                'generator' :rand,
                'type' :'percentage',
                'percentage' :  percentage,}


    def add_generator_mean(self, title, mean, sd):
        """
        Add a random float generator based on mean value and standard deviation

        :param title: name of generator
        :param mean: mean value to to generate numbers around
        :param sd: standard deviation
        """

        if isinstance(title,str):
            rand = np.random
            seed = self.generate_seed()
            rand.seed(seed)
            self.generators[title] = {
                'generator' : rand,
                'type' : 'mean',
                'mean' : mean,
                'sd' : sd}

    def add_generator_poisson(self, title, mean):
        """
        Add a random float generator based on poisson distribution

        :param title: name of generator
        :param mean:  mean value to to generate numbers around
        """

        if isinstance(title,str):
            rand = np.random
            seed = self.generate_seed()
            rand.seed(seed)
            self.generators[title] = {
                'generator' : rand,
                'type' : 'poisson',
                'mean' : mean}

    def add_generator_bit(self, title, bit_value):
        """
        Add a random int generator based on bit value

        :param title: name of generator
        :param bit_value:  bit value to generate value from
        """

        if isinstance(title,str):
            rand = random.Random()
            seed = self.generate_seed()
            rand.seed(seed)
            self.generators[title] = {
                'generator': rand,
                'type': 'bit',
                'bit_value': bit_value}

    def add_generator_log(self, title, mean, sigma):
        """
        Add a random int generator based on bit value

        :param title: name of generator
        :param bit_value:  bit value to generate value from
        """

        if isinstance(title,str):
            rand = np.random
            seed = self.generate_seed()
            rand.seed(seed)
            self.generators[title] = {
                'generator': rand,
                'type': 'log',
                'mean': mean,
                'sigma' : sigma}


    def __getitem__(self, title):
        """

        return the random number for a given generator

        :param title: name of generator
        :return: random number from the generator
        """

        if title in self.generators.keys() and isinstance(title, str):
            if self.generators[title]['type'] == 'int':
                return self.generators[title]['generator'].randint(self.generators[title]['start'],
                                                                   self.generators[title]['end'])
            elif self.generators[title]['type'] == 'mean':
                return self.generators[title]['generator'].normal(self.generators[title]['mean'],
                                                                  self.generators[title]['sd'])
            elif self.generators[title]['type'] == 'poisson':
                return self.generators[title]['generator'].poisson(self.generators[title]['mean'])
            elif self.generators[title]['type'] == 'bit':
                return self.generators[title]['generator'].getrandbits(self.generators[title]['bit_value'])

            elif self.generators[title]['type'] == 'percentage':
                return (self.generators[title]['generator'].randint(0, 100) <= self.generators[title]['percentage'])

            elif self.generators[title]['type'] == 'log':
                return self.generators[title]['generator'].lognormal(self.generators[title]['mean'],
                                                                      self.generators[title]['sigma'])



    def pull_multiple(self, title, num):
        """
        Pulls multiple random numbers from a generator

        :param title: name of generator
        :param num: number of random number to pull
        :return: list of random numbers
        """

        if title in self.generators.keys() and isinstance(title, str):
            l = []
            if self.generators[title]['type'] == 'int':
                for i in range(num):
                    l.append(self.generators[title]['generator'].randint(self.generators[title]['start'],self.generators[title]['end']))
            elif self.generators[title]['type'] == 'mean':
                for i in range(num):
                    l.append(self.generators[title]['generator'].normal(self.generators[title]['mean'],self.generators[title]['sd']))
            elif self.generators[title]['type'] == 'poisson':
                for i in range(num):
                    l.append(self.generators[title]['generator'].poisson(self.generators[title]['mean']) )
            elif self.generators[title]['type'] == 'bit':
                for i in range(num):
                    l.append(self.generators[title]['generator'].getrandbits(self.generators[title]['bit_value']))


            return l



class Random_Generator_Tests(unittest.TestCase):

    def setUp(self):
        pass

    def test_seed(self):

        rgen = Random_generator(123456)
        rgen.add_generator_int('stop_time', 0, 300)
        rgen.add_generator_int('distance', 300, 600)
        st = []
        d = []
        for i in range(10):
            st.append(rgen['stop_time'])
            d.append(rgen['distance'])

        rgen2 = Random_generator(123456)
        rgen2.add_generator_int('stop_time', 0, 300)
        rgen2.add_generator_int('distance', 300, 600)
        nst = []
        nd = []
        for i in range(10):
            nst.append(rgen2['stop_time'])
            nd.append(rgen2['distance'])

        assert nst == st
        assert nd == d

    def test_random(self):
        rgen = Random_generator()
        rgen.add_generator_int('xxxx')

        l = []
        for i in range(1000):
            l.append(rgen['xxxx'])
        assert max(l)<101
        assert min(l)>-1

    def test_random_mean(self):
        rgen = Random_generator(123456)
        rgen.add_generator_mean('xxx', 50, 1)

        v1 = rgen['xxx']
        v2 = rgen['xxx']

        assert int(v1) == 51

        assert v1!=v2

        rgen2 = Random_generator(123456)
        rgen2.add_generator_mean('xxx', 50, 1)
        v3 = rgen2['xxx']

        assert v1==v3

    def test_random_poisson(self):
        rgen = Random_generator(123456)
        rgen.add_generator_poisson('xxxx', 20)
        v1 = rgen['xxxx']
        v2 = rgen['xxxx']

        assert v1 == 20 #based on seed value
        assert v1!=v2

        rgen2 = Random_generator(123456)
        rgen2.add_generator_poisson('xxx', 20)
        v3 = rgen2['xxx']

        assert v1==v3

    def test_multiple(self):
        rgen = Random_generator(123456)
        rgen.add_generator_poisson('xxxx', 20)
        v2 = rgen.pull_multiple('xxxx', 10)

        assert len(v2) == 10

        rgen2 = Random_generator(123456)
        rgen2.add_generator_poisson('xxx', 20)
        v3 = rgen.pull_multiple('xxxx', 10)

        assert v2==v3

    def test_generate_seed(self):
        rgen = Random_generator(123456)
        seeds = []
        for i in xrange(25):
          seeds.append(rgen.generate_seed())

        rgen2 = Random_generator(123456)
        seeds2 = []
        for i in xrange(25):
          seeds2.append(rgen2.generate_seed())

        assert  seeds == seeds2

    def test_random_bit(self):
        rgen = Random_generator(123456)
        rgen.add_generator_bit('xxxx', 4)
        v1 = rgen['xxxx']
        v2 = rgen['xxxx']

        assert v1 == 14 #based on seed value
        assert v1!=v2

        rgen2 = Random_generator(123456)
        rgen2.add_generator_bit('xxx', 4)
        v3 = rgen2['xxx']

        assert v1==v3

    def test_random_percentage(self):
        rgen = Random_generator(123456)
        rgen.add_generator_percentage('xxxx', 50)
        v1 = rgen['xxxx']
        v2 = rgen['xxxx']

        assert v1 == False #based on seed value
        assert v1==v2

        rgen2 = Random_generator(123456)
        rgen2.add_generator_percentage('xxx', 50)
        v3 = rgen2['xxx']

        assert v1==v3

    def test_random_log(self):
        rgen = Random_generator(123456)
        rgen.add_generator_log('xxxx', 0, 1)
        v1 = rgen['xxxx']
        v2 = rgen['xxxx']

        assert round(v1, 5) == 3.68449 #based on seed value
        assert v1!=v2

        rgen2 = Random_generator(123456)
        rgen2.add_generator_log('xxx', 0, 1)
        v3 = rgen2['xxx']

        assert v1==v3



if __name__ == '__main__':
    unittest.main()






