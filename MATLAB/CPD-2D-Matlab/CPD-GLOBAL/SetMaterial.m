function [ out ] = SetMaterial( inp , L , Delta , MatPars , MatLaw ,  flag )

	switch flag

	case 'PL'

		PL = inp;

		if ( isobject(PL(1)) )
			NoPs = size(PL,2);
		else
			NoPs = 0;
		end

		for p = 1:NoPs

			mat = 1;

			PL(p).L = L;
			PL(p).Delta = Delta;
			PL(p).Mat = mat;
			PL(p).MatPars = MatPars(mat,:);
			PL(p).MatLaw = MatLaw;

		end

		out = PL;

	case 'EL'

		EL = inp;
		NoEs = size(EL,2);

		for e = 1:NoEs;

			mat = 1;

			EL(e).Mat = mat;
			EL(e).MatPars = MatPars(mat,:);
			EL(e).MatLaw = MatLaw;

		end

		out = EL;

	end

end

