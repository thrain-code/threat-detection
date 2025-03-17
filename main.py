import numpy as np

a = input("angka kesatu")
b = input("angka kedua")

nilai = [76,50.80,97,40,60]

c = int(a) + int(b)

if c > np.mean(nilai):
  print(c, "anda lulus nilai rata rata nya adalah ", np.mean(nilai))
else :
  print(c, "anda gagal nilai rata rata nya adalah ", np.mean(nilai))