uses GraphABC;
const N=50;
Var i,k,m,s,flag : integer;
 x1,x2,x4,x5,x6,q,lenta: string;
 kom,a : array[1..N] of string; Label m1;
BEGIN lenta:='_ABBAABAB_______';
m:=1; q:= '1';
kom[1]:='1_>1_R'; kom[2]:='2_>3|R';
kom[3]:='3_>4AL'; kom[4]:= '4_>1_R';
kom[5]:='1A>2*R'; kom[6]:='2A>2AR';
kom[7]:='3A>3AR'; kom[8]:='4A>4AL';
kom[9]:='1B>1BR'; kom[10]:='2B>2BR';
kom[11]:='4B>4BL'; kom[12]:='1*>1*R'; 
kom[13]:='2*>2*R'; kom[14]:='4*>4*L';
kom[15]:='1|>1|S'; kom[16]:='2|>3|R';
kom[17]:='4|>4|L';
For i:=1 to N do a[i]:=copy(lenta,i,1);
Repeat flag:=0; s:=s+1;
 For i:=1 to N do begin
 x1:=copy(kom[i],1,1); x2:=copy(kom[i],2,1);
 x4:=copy(kom[i],4,1); x5:=copy(kom[i],5,1);
 x6:=copy(kom[i],6,1);
 If (flag=0)and(x1=q)and(x2=a[m]) then
 begin q:=x4; a[m]:=x5;
 If x6='R' then m:=m+1;
 If x6='L' then m:=m-1;
 If x6='S' then goto m1;
 flag:=1; end; end;
 m1: k:=k+1;
For i:=1 to 25 do write(a[i],' ');
writeln(' ',q,' k=',k); Sleep(5);
For i:=1 to m-1 do write('=='); write('|'); writeln;
until x6='S';
END.
