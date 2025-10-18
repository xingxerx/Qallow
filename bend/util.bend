# Utility helpers shared across the Bend translation of Qallow.

def clamp(x, lo, hi):
  if x < lo then lo
  elif x > hi then hi
  else x

def clamp_unit(x):
  clamp(x, 0.0, 1.0)

def clamp_entropy(x):
  clamp(x, 0.0, 0.001)

def clamp_coherence(x):
  clamp(x, 0.0, 1.0)

def clamp_decoherence(x):
  clamp(x, 0.0, 0.001)

def abs(x):
  if x < 0.0 then -x else x

def lerp(a, b, t):
  a + (b - a) * t

def str(x):
  show(x)

def strf(x, _precision):
  show(x)

def PI():
  3.141592653589793

def to_float(x):
  x
