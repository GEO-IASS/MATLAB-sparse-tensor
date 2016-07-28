A = rand(2,2,2,2,2);
B = A


Us = sparseTensor(A);
Us.permute([1,2,3,5,4]);
As = sparseTensor(A);
As.permute([2,1,3,4,5]);
Bs = sparseTensor(B);
Bs.permute([2,1,3,4,5]);
for ind1 = 2:2:sL-1
    Bs.permute([2,3,4,5,1]);
    Us = Us.fold(Bs);
    Us.permute([1,5,2,3,6,4,8,7]); 
    Bs.permute([5,1,2,3,4]);
    sU = Us.size();
    Us.reshape([sU(1)*sU(2),sU(3),sU(4)*sU(5),sU(6)*sU(7),sU(8)]);    
    As.permute([2,3,4,5,1]);
    Us = Us.fold(As);
    As.permute([5,1,2,3,4]);
    Us.permute([1,5,2,3,6,4,8,7]);
    sU = Us.size();
    Us.reshape([sU(1)*sU(2),sU(3),sU(4)*sU(5),sU(6)*sU(7),sU(8)]);
end
Us.permute([1,3,4,2,5]); %%% ?????? 2-5
Bs.permute([1,4,2,3,5]);
sU = Us.size();
sB = Bs.size();
Us.reshape([sU(1),sU(2),sU(3),sU(4)*sU(5)]);
Bs.reshape([sB(1)*sB(2),sB(3),sB(4),sB(5)]);
Bs.permute([2,3,4,1]);
Us = Us.fold(Bs);
Us.permute([1,4,2,5,3,6]);
sU = Us.size();   
Us.reshape([sU(1)*sU(2),sU(3)*sU(4),sU(5)*sU(6)]);

sU = Us.size()
Us.reshape([sU(1),sU(2)*sU(3)]);
Us.size()

