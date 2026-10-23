# The authors test edge set from their split files, in their snapshot alignment (from the test
# start, 150885) and in AuthBench alignment (multiples of 1800 s), self-loops skipped as in load_lanl.py.
BEGIN { FS = ","; S = 150885; D = 1800 }
{
  ts = $1 + 0
  if (ts < S || $2 == $3) next
  a = int((ts - S) / D); b = int(ts / D)
  A[a "," $2 "," $3] = 1; B[b "," $2 "," $3] = 1
  if (a > maxa) maxa = a
  fixed = ($4 + 0 == 1) || (ts == 150885 && $2 == 9660 && $3 == 44)
  if ($4 + 0 == 1) { AA[a "," $2 "," $3] = 1; BB[b "," $2 "," $3] = 1 }
  if (fixed) { AF[a "," $2 "," $3] = 1; BF[b "," $2 "," $3] = 1 }
}
END {
  for (k in A) { na++; split(k, p, ","); if (p[1] == maxa) last++ }
  for (k in B) nb++
  for (k in AA) { naa++; split(k, p, ","); if (p[1] == maxa) lasta++ }
  for (k in BB) nbb++
  for (k in AF) naf++
  for (k in BF) nbf++
  printf "authors alignment: %d edges (%d in the last snapshot, dropped by the loader), anomalous %d (%d in last), with first red-team line labelled %d\n", na, last, naa, lasta, naf
  printf "AuthBench alignment: %d edges, anomalous %d, with first red-team line labelled %d\n", nb, nbb, nbf
}
