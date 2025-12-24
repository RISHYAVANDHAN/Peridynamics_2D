function out = MainCPD( PD , SiZe , L , Delta , steps , d , BCflag , DEFflag , TOPflag , C1 , C2 , C3 , MATlaw )
    clc;
	% ACflaf is 0 (without) or 1 (with) area correction ... 

	% L          : lattice length in the material configuration
	% Delta      : horizon size in the material configuration 
	% steps      : number of increments
	% d          : prescribed displacement
	% BCflag     : 'DBC' , 'STD' , 'XTD'
	% DEFflag    : 'EXT' , 'SHR' , 'EXP'
	% TOPflag    : topology ... without or with hole ... 'FULL' , 'PART' , 'CUT'

	Geometry = 'tetragone-uniform';

	tic

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%       ADD PATH        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	addpath './../CPD-OBJECTS/';
	addpath './../CPD-GLOBAL/';

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%       PROBLEM DEFINITION        %%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	Corners = Compute_Corners(PD,SiZe);

	PatchFlag = Compute_PatchFlag(BCflag,DEFflag);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%     BULK PROPERTIES     %%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	radius = 0.2*SiZe;

	center = zeros(1,PD);

	Bvals = [ radius , center ]; % radius and center of the void

	mu = 0; % dummy
	lambda = 0; % dummy

	BMATparsCCM(1,:) = [ mu , lambda ];
	BMATparsCPD(1,:) = [ C1 , C2 , C3 ];

	% MATlaw = 'SS'; % 'AJ' , 'SS'

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%       CREATE MESH        %%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[ NLtmp , CNCT ] = Mesh( Corners , L );
	
	% Plot( NLtmp , 'mesh' , CNCT );
	% Plot( NLtmp , 'nodes' );

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%       CREATE PATCH        %%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% patch is created separately since it must always be uniform/structured
	% also to separate "patch-flag" from the mesh discretization

	[ NLext ] = Patch( Corners , L , Delta , PatchFlag );
	% [ NLext ] = Patch( Corners , L , Delta , 'fullpatch' );

	NL = [ NLtmp ; NLext ];

	% Plot( NL , 'nodes' );

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%       CREATE TOPOLOGY        %%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[ PL ] = Topology( NL , CNCT , L , Delta , Bvals , TOPflag );

	% Plot( PL , 'points' );

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%       NEIGHBORLIST ASSIGNMENT TO POINTS        %%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[ PL ] = AssignNgbrs( PL , L , Delta );

	[ PL ] = AssignVols( Corners , PL , L , Bvals , TOPflag ); 

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%       SET MATERIAL        %%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[ PL ] = SetMaterial( PL , L , Delta , BMATparsCPD , MATlaw , 'PL' );

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%       INFO OUTPUT        %%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	NoNs = size(NL,1);
	NoPs = size(PL,2);

    disp(['======================================================']);
	disp(['number of nodes                 : ' , num2str(NoNs) ]);
	disp(['number of points                : ' , num2str(NoPs) ]);
    disp(['======================================================']);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%      OVERALL DEFORMATION      %%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	FF = Compute_FF(PD,d,DEFflag);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%      ASSIGN BOUNDARY CONDITIONS      %%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[ PL , DOFs ] = AssignBCs( Corners , PL , FF , BCflag , PatchFlag );

	disp(['======================================================']);
	disp(['number of DOFs                  : ' , num2str(DOFs) ]);
	disp(['======================================================']);

	% PointListInfo( PL );

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%      INPUT PARAMETERS      %%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	time = datetime;

			  % '@' num2str(time.Hour) '_' num2str(time.Minute)...
			  % '@' num2str(time.Day) '.' num2str(time.Month) '.' num2str(time.Year) ...
	% Outcore = ['/Users/Ali/SIM-LOCAL/' num2str(PD) 'D_' DEFflag '_' BCflag '_' TOPflag ...
	Outcore = [ num2str(PD) 'D_' DEFflag '_' BCflag '_' TOPflag ...
			  '_L_' num2str(L) '_Delta_' num2str(Delta) '_' MATlaw ... 
			  '_C1_' num2str(C1) '_C2_' num2str(C2) '_C3_' num2str(C3) ];

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%        COMPUTATION         %%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     et = toc;
    
	filename = [Outcore '.txt'];
    file = fopen(filename,'wt');

    fprintf(file,'\n======================================================\n');
    fprintf(file,'\nnumber of DOFs                  : %1.3d\n',DOFs);
    fprintf(file,'\nnumber of POINTs                : %1.3d\n',size(PL,2));
    fprintf(file,'\n======================================================\n');
%     fprintf(file,'\nelapsed time: %1.3d seconds\n',et);

    loadStep = 1/steps;

    min_try = 0;
    max_try = 20;

    counter=0;

    LF = 0;

    tol = 1e-11;

    while ( LF <= 1.0 + 1e-8 )
        
        disp([char(10) 'Load factor : ' num2str(LF)]);
        fprintf(file,[char(10) 'Load factor : ' num2str(LF) '\n']);

        [ PL ] = Update( PL , LF , 'prescribed' );

        error_counter=1;
        isNotAccurate=true;
        while (isNotAccurate)

            R = Assembly( PL , Delta , DOFs , LF , 'residual' );

            if (error_counter == 1)
                normnull = norm(R);
                fprintf('Residual Norm @ Increment %u @ Iteration %u : %1.2d    ,    normalized : 1\n',counter,error_counter,norm(R));
                fprintf(file,'Residual Norm @ Increment %u @ Iteration %u : %1.2d    ,    normalized : 1\n',counter,error_counter,norm(R));
            else
                fprintf('Residual Norm @ Increment %u @ Iteration %u : %1.2d    ,    normalized : %1.2d\n',counter,error_counter,norm(R),norm(R)/normnull);
                fprintf(file,'Residual Norm @ Increment %u @ Iteration %u : %1.2d    ,    normalized : %1.2d\n',counter,error_counter,norm(R),norm(R)/normnull);
            end

            if ( error_counter>1 )

                if ((((norm(R)/normnull)<tol || norm(R)<tol) || error_counter>max_try) && error_counter>min_try)
                    isNotAccurate = false;
                    break;
                end

            end

            K = Assembly( PL , Delta , DOFs , LF , 'stiffness' );
            
            A=full(K);
            
            dx = - K \ R;
            
            KKtmp = K - K';
            
%             disp(['nonsymmetry measure : ' num2str(norm( full(KKtmp) ))]);
                        
            [ PL ] = Update( PL , dx , 'displacement' );
            
            error_counter = error_counter+1;

        end

%         et = toc;
%         fprintf(file,'\nelapsed time: %1.3d seconds\n',et);

        if (error_counter>max_try)
            disp('Convergence is not obtained!')
            fprintf(file,'Convergence is not obtained!');
            break;
        end

        % Volumes = Assembly( PL , Delta , DOFs , LF , 'point-volume' )

        % Volume = Assembly( PL , Delta , DOFs , LF , 'volume' );

        % Psi = Assembly( PL , Delta , DOFs , LF , 'energy' );

        % Psi_Anlaytical = compute_Psi_analytical( PD , L , Delta , SiZe , d , C1 , C2 , C3, MATlaw );

        counter = counter+1;

        % PostProcess( counter , PL , EL , Outcore , 'write-output-ParaView-points');
        % PostProcess( counter , PL , EL , Outcore , 'write-output-ParaView-elements');
        % nu = PostProcess( counter , PL , EL , Outcore , 'poisson');
     
        LF = LF + loadStep;

    end

%     PostProcess( counter , PL , EL , Outcore , 'write-output-graph-data');

    % out = 0;
    % out = [Psi , Psi_Anlaytical];
    % out = nu;

    % fprintf('for delta/Delta : %1.6d    ,    nu = %1.6d\n',((Delta-0.0001)/L),nu);
    % fprintf(file,'for delta/Delta : %1.6d    ,    nu = %1.6d\n',((Delta-0.0001)/L),nu);

    toc

    % figure1 = figure;
    % spy(K)
    % filename = [Outcore '.png'];
    % saveas(figure1,filename)

    fclose(file);

    % full(K)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%       REMOVER PATH        %%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	rmpath './../CPD-OBJECTS/';
	rmpath './../CPD-GLOBAL/';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         FUNCTIONS         %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = compute_Psi_analytical( PD , L , Delta, SiZe , d , C1 , C2 , C3 , MATlaw )

	out = 0;

	if ( PD==2 )

		l = L * (1+d);
		S1 = (l-L)/L;
		S2 = (l^2-L^2)/L^2;

		switch MATlaw

	    case 'SS'

			out1 = 1/3 * C1 * pi * S1^2 * Delta^3 * SiZe^2;
			out2 = 1/4 * C2 * pi^2 * S2^2 * Delta^4 * SiZe^2;

	    case 'AJ'

	    	A = pi * Delta^2;

			out1 = 1/2 * C1 * S1^2 * A * SiZe^2;
			out2 = 1/2 * C2 * S2^2 * A^2 * SiZe^2;

	    end

		out = out1 + out2;

	elseif ( PD==3 )

		l = L * (1+d);
		S1 = (l-L)/L;
		S2 = (l^2-L^2)/L^2;
		S3 = (l^3-L^3)/L^3;

		switch MATlaw

	    case 'SS'

			% out1 = 1/3 * C1 * pi * S1^2 * Delta^3 * SiZe^2;
			% out2 = 1/4 * C2 * pi^2 * S2^2 * Delta^4 * SiZe^2;
			% out3 = 1/4 * C2 * pi^2 * S3^2 * Delta^4 * SiZe^2;

	    case 'AJ'

			V = 4/3 * pi * Delta^3;

			out1 = 1/2 * C1 * S1^2 * V * SiZe^2;
			out2 = 1/2 * C2 * S2^2 * V^2 * SiZe^2;
			out3 = 1/2 * C3 * S3^2 * V^3 * SiZe^2;

	    end

		out = out1 + out2 + out3;

	end

end

function PatchFlag = Compute_PatchFlag(BCflag,DEFflag)

	if ( BCflag == 'DBC' )

		PatchFlag = 'fullpatch'; % 'fullpatch'  or  'horzpatch'   or  'vertpatch'

	elseif ( BCflag == 'STD' )

		PatchFlag = 'horzpatch'; % 'fullpatch'  or  'horzpatch'   or  'vertpatch'

	elseif ( BCflag == 'XTD' ) % 

		PatchFlag = 'fullpatch'; % 'fullpatch'  or  'horzpatch'   or  'vertpatch'

	elseif ( BCflag == 'XTM' ) % 

		PatchFlag = 'fullpatch'; % 'fullpatch'  or  'horzpatch'   or  'vertpatch'

	end

end

function Corners = Compute_Corners(PD,SiZe)

	if ( PD == 2 )

		Corners(1,:) = SiZe * 1/2 * [ -1 , -1 ]; % left south corner
		Corners(2,:) = SiZe * 1/2 * [ 1 , -1 ]; % right south corner
		Corners(3,:) = SiZe * 1/2 * [ 1 , 1 ]; % right north corner
		Corners(4,:) = SiZe * 1/2 * [ -1 , 1 ]; % left north corner

	elseif ( PD == 3 )

		Corners(1,:) = SiZe * 1/2 * [ -1 , -1 , -1 ]; % left south bottom corner
		Corners(2,:) = SiZe * 1/2 * [ 1 , -1 , -1 ]; % right south bottom corner
		Corners(3,:) = SiZe * 1/2 * [ 1 , 1 , -1 ]; % right north bottom corner
		Corners(4,:) = SiZe * 1/2 * [ -1 , 1 , -1 ]; % left north bottom corner
		Corners(5,:) = SiZe * 1/2 * [ -1 , -1 , 1 ]; % left south top corner
		Corners(6,:) = SiZe * 1/2 * [ 1 , -1 , 1 ]; % right south top corner
		Corners(7,:) = SiZe * 1/2 * [ 1 , 1 , 1 ]; % right north top corner
		Corners(8,:) = SiZe * 1/2 * [ -1 , 1 , 1 ]; % left north top corner

	end

end

function FF = Compute_FF(PD,d,DEFflag)

	I = eye(PD,PD);

	switch DEFflag
	case 'EXT'
	    FF = I;
	    FF(1,1) = 1 + d;
	case 'EXP'
	    FF = (1+d) * I;
	case 'SHR'
	    FF = I;
	    FF(2,1) = d;
	end

end
