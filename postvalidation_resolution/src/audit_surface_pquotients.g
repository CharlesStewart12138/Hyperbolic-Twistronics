LoadPackage("anupq");;
F := FreeGroup("a","b","c","d");;
a := F.1;; b := F.2;; c := F.3;; d := F.4;;
G := F / [a*b^-1*c*d^-1*a^-1*b*c^-1*d];;
for p in [2,3] do
  for class in [1,2,3] do
    started := Runtime();;
    epi := PqEpimorphism(G : Prime := p, ClassBound := class);;
    Q := Image(epi);;
    Print("P=",p," CLASS=",class," ORDER=",Size(Q)," PCGENS=",Length( pcgs(Q) ),
          " PCLASS=",PClassPGroup(Q)," RUNTIME_MS=",Runtime()-started,"\n");
    Print("IMAGES=",List(GeneratorsOfGroup(G),x->Image(epi,x)),"\n");
  od;
od;
QUIT;
