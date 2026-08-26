LoadPackage("kbmag");;
outfile := GAPInfo.SystemEnvironment.VALIDATION_NORMAL_FORMS;;
maxlen := Int(GAPInfo.SystemEnvironment.VALIDATION_MAX_LENGTH);;
F := FreeGroup("a","b","c","d");;
a := F.1;; b := F.2;; c := F.3;; d := F.4;;
G := F / [a*b^-1*c*d^-1*a^-1*b*c^-1*d];;
R := KBMAGRewritingSystem(G);;
ok := AutomaticStructure(R);;
if not ok then
  Error("KBMAG automatic structure failed");
fi;
words := EnumerateReducedWords(R,0,maxlen);;
out := OutputTextFile(outfile,false);;
for w in words do
  AppendTo(out, ExtRepOfObj(w), "\n");
od;
CloseStream(out);;
Print("GAP_VERSION=", GAPInfo.Version, "\n");
Print("AUTOMATIC=true\n");
Print("GROWTH=", GrowthFunction(R), "\n");
Print("NORMAL_FORM_COUNT=", Length(words), "\n");
QUIT;

