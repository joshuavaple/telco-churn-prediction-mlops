import uuid
import random


def generate_uuid4():
  return uuid.uuid4().hex

def generate_random_int():
  """a function to generat a random interger from 1-999"""
  return random.randint(1, 999)