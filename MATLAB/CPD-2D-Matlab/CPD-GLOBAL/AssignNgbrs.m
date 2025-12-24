function [ PL ] = AssignNgbrs( PL , L , Delta )

	NoPs = size(PL,2);

	PD = PL(1).PD;

	tol = 1e-8;

	NmaxNgbr = 0;

	Del_by_L = floor(Delta/L);

	if ( PD==2 )

		for i = -Del_by_L : Del_by_L

			for j = -Del_by_L : Del_by_L

				if ( ( sqrt(i^2+j^2) * L < Delta ) && ( i~=0 || j~=0 ) )

			    	NmaxNgbr = NmaxNgbr + 1;

			    end

			end

		end

	elseif ( PD==3 )

    	for i = -Del_by_L : Del_by_L

			for j = -Del_by_L : Del_by_L

				for k = -Del_by_L : Del_by_L

					if ( ( sqrt(i^2+j^2+k^2) * L < Delta ) && ( i~=0 || j~=0 || k~=0 ) )

				    	NmaxNgbr = NmaxNgbr + 1;

				    end

				end

			end

		end

    end

	for p = 1 : NoPs

		nc = 0;

		for q = 1 : NoPs

			if ( norm( PL(p).X - PL(q).X ) < Delta )

				if ( q ~= p )

					nc = nc + 1;
					neighbors(nc) = q;
					neighborsX(:,nc) = PL(q).X;
					neighborsx(:,nc) = PL(q).x;

				end

			end

		end

		PL(p).neighbors = neighbors;
		PL(p).neighborsx = neighborsx;
		PL(p).neighborsX = neighborsX;

		X = PL(p).X;

		NNgbr = length(neighbors);

		NInII = 0;

		if ( PD==2 )

			for i = 1 : NNgbr

		        for j = 1 : NNgbr

		            if ( j ~= i )

		                XiI = neighborsX(:,i) - X;
		                XiII = neighborsX(:,j) - X;

		                A = cross([XiI' 0] , [XiII' 0]);

		                if ( norm(A) > tol && norm(XiI-XiII) < Delta )

		                	NInII = NInII + 1;

		                end

		            end

		        end

		    end

		elseif ( PD==3 )

			for i = 1 : NNgbr

		        for j = 1 : NNgbr

		            if ( j ~= i )

		                XiI = neighborsX(:,i) - X;
		                XiII = neighborsX(:,j) - X;

		                A = cross(XiI' , XiII');

	                	if ( norm(A) > tol && norm(XiI-XiII) < Delta )

		                	NInII = NInII + 1;

		                end

		            end

		        end

		    end

		end

	    NInIInIII = 0;

	    if ( PD==3 )

			for i = 1 : NNgbr

	            for j = 1 : NNgbr

	                if ( j ~= i )

	                    for k = 1 : NNgbr

	                        if ( (k ~= i) && (k ~= j) )

	                            XiI = neighborsX(:,i) - X;
	                            XiII = neighborsX(:,j) - X;
	                            XiIII = neighborsX(:,k) - X;

	                            V = XiI'*(cross(XiII',XiIII'))';

			                	if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

				                	NInIInIII = NInIInIII + 1;

				                end

	                        end

	                    end

	                end

	            end

	        end

	    end

	    if ( PD==2 )

	    	Amax = pi * Delta^2;

	    	AV = (NNgbr+1)/(NmaxNgbr+1) * Amax;

	    elseif ( PD==3 )

	    	Vmax = 4/3 * pi * Delta^3;

	    	AV = (NNgbr+1)/(NmaxNgbr+1) * Vmax;

	    end
		
		PL(p).NI = NNgbr;
		PL(p).NInII = NInII;
		PL(p).NInIInIII = NInIInIII;
		PL(p).AV = AV;

		neighbors = 0;
		neighborsx(:,:) = 0;
		neighborsX(:,:) = 0;

	end

end
