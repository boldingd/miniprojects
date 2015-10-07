#! /usr/bin/python3

import sys
import random
import string

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--length", type=int, help="the length of the password to generate")
args = parser.parse_args()

pw_length = 25
if args.length is not None:
  pw_length = args.length

password_chars = [ ch for ch in string.printable if ch not in string.whitespace ]

pw_array = [ random.choice(password_chars) for _ in range(pw_length) ]
pw_string = ''.join(pw_array)

print(pw_string)

