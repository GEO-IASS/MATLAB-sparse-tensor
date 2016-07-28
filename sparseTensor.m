classdef sparseTensor < handle
    
    properties (SetAccess = private)
        totalS % total size of the tensor
        A % tensor data
        indices % list with the size of each dimension
    end
    
    methods    
        % creation method
        function st = sparseTensor(A)
            sA = size(A);
            st.totalS = prod(sA);
            st.indices = sA;
            st.A = reshape(A,[st.totalS,1]);
            if ~issparse(A)
                st.A = sparse(st.A);
            end
        end
        
        % dimension sizes
        function U = size(st)
            U =  st.indices;
        end
        
        % total number of dimensions
        function U = ndims(st)
            U = length(st.indices);
        end
        
        % reshape the tensor
        function reshape(st,indices)
            if prod(indices) == st.totalS
                st.indices = indices;
            else
                error('Reshape: Cannot reshape, dimension missmatch');
            end
        end
        
        % compute indices 
        function U = iN(st,s,order)
            div = [];
            findices = fliplr(st.indices);
            s = s - 1;
            
            for ind1 = 1:length(findices)
                base = findices(end-ind1+1);
                div = [mod(s,base) div];
                %fprintf(1,'index: %d base: %d rem: %d\n',mod(s,base), base, s);
                s = floor(s/base);
            end
            div = fliplr(div);
            findices = st.indices(order);
            div = div(order);
            U = 1 + div(1);
            base = findices(1);
            for ind1 = 2:length(findices)
                U = U + base*div(ind1);
                base = base*findices(ind1);
            end
        end
        
        % permute some indices
        function permute(st,order)
            if st.ndims() == length(order)
                st.A = reshape(st.A,[st.totalS 1]);
                [i,~,s] = find(st.A);
                An = spalloc(st.totalS,1,length(s));
                for ind1 = 1:length(s)
                    %st.iN(i(ind1),order);
                    n = st.iN(i(ind1),order);
                    An(n,1) = s(ind1);
                end
                st.A = An;
                st.indices = st.indices(order);
            else
                error('Permute: Number of dimensions must be equal');
            end
        end
        
        % returns a full version of the sparse tensor
        function U = getFull(st)
            U = full(st.A);
            U = reshape(U,st.indices);
        end
        
        % folds two sparse tensors along the last index of both tensors
        function U = fold(st,B)
            nA = reshape(st.A,[prod(st.indices(1:end-1)),st.indices(end)]);
            nB = reshape(B.A,[prod(B.indices(1:end-1)),B.indices(end)]);
            if st.indices(end) ~= B.indices(end)
                error('Fold: size indexes do not match');
            end
            C = nA*nB';
            U = sparseTensor(C);
            U.reshape([st.indices(1:end-1) B.indices(1:end-1)]);
        end
       
        
            
    end
end
