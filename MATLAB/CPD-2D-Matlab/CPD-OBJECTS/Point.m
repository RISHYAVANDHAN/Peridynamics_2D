classdef Point

    % POINT class:
    % Point(Nr,X)
    % 
    % Nr           scalar                             point number
    % NNr          scalar                             corresponding node number
    % X            vector [PD X 1]                    reference coordinates
    % x            vector [PD X 1]                    current coordinates
    % BCflg        vector [PD X 1]                    flag for boundary condition ... 0 : Dirichlet ... 1 : Neumann
    % BCval        vector [PD X 1]                    corresponding value for the condition ... displacement for Dirichlet ... force for Neumann
    % DOF          vector [PD X 1]                    global degree of freedom
    % BCset        scalar                             boolean to check if the global DOF is set ... 0 : False (default) ... 1 : True (DOF is set)
    % PD           scalar                             problem dimension
    % neighbors    vector [1 X NNgbr]                 neighbors of the point within the horizon
    % neighborsx   matrix [PD X NNgbr]                current coordinates of neighbors of the point within the horizon
    % neighborsX   matrix [PD X NNgbr]                reference coordinates of neighbors of the point within the horizon
    % NI           scalar                             number of 1st neighbors
    % NInII        scalar                             number of 1st and 2nd neighbor combinations forming non-zero areas
    % NInIInIII    scalar                             number of 1st, 2nd and 3rd neighbor combinations forming non-zero volumes
    % Mat          scalar                             material identifier
    % MatPars      vector [1 X NPar]                  material parameters
    % MatLaw       string                             constitutive law
    % Delta        scalar                             horizon size
    % L            scalar                             lattice length
    % Vol          scalar                             volume of the point
    % AV           scalar                             area or volume of the point in the peridynamic sense 

    properties
    	Nr     
        NNr     
    	X      
    	x      
    	BCflg  
    	BCval
        BCset
        DOF    
        PD   
        neighbors
        neighborsX
        neighborsx
        NI
        NInII
        NInIInIII
        AV
        Mat
        MatPars
        MatLaw
        Delta
        L
        Vol
    end

    methods

    	function obj = Point(Nr,X)
    		obj.PD = size(X,1);
			obj.Nr = Nr;
			obj.X = X;
			obj.x = X;
			obj.BCflg = zeros(obj.PD,1);
    		obj.BCval = zeros(obj.PD,1);
            obj.BCset = 0;
            obj.DOF = zeros(obj.PD,1);
            obj.neighbors = 0;
            obj.NI = 0;
            obj.NInII = 0;
            obj.NInIInIII = 0;
            obj.AV = 0;
            obj.Vol = 0;
	    end

        function obj = set.NNr(obj,NNr)
            obj.NNr = NNr;
        end

        function obj = set.Mat(obj,Mat)
            obj.Mat = Mat;
        end

        function obj = set.MatPars(obj,MatPars)
            obj.MatPars = MatPars;
        end

        function obj = set.MatLaw(obj,MatLaw)
            obj.MatLaw = MatLaw;
        end

        function obj = set.Delta(obj,Delta)
            obj.Delta = Delta;
        end

        function obj = set.L(obj,L)
            obj.L = L;
        end

        function obj = set.neighbors(obj,neighbors)
            obj.neighbors = neighbors;
        end

        function obj = set.neighborsX(obj,neighborsX)
            obj.neighborsX = neighborsX;
        end

        function obj = set.neighborsx(obj,neighborsx)
            obj.neighborsx = neighborsx;
        end

        function out = eq(N1,N2) % override == operator
            tol = 1.0e-6;
            X1 = N1.X;
            X2 = N2.X;
            out = true;
            for i = 1 : obj.PD
                if ( abs(X1(i)-X2(i))>tol )
                    out = false;
                end
            end
        end

        function out = Info(obj)

            disp(['======================================================']);
            disp(['number                : ' , num2str(obj.Nr) ]);
            disp(['X                     : ' , num2str((obj.X)') ]);
            disp(['x                     : ' , num2str((obj.x)') ]);
            disp(['BCflag                : ' , num2str((obj.BCflg)') ]);
            disp(['BCval                 : ' , num2str((obj.BCval)') ]);
            disp(['DOFs                  : ' , num2str((obj.DOF)') ]);
            disp(['neighbors             : ' , num2str((obj.neighbors)) ]);
            disp(['======================================================']);
                 
        end

        function vols = volumes(obj)
            vols = compute_volumes(obj.Nr,obj.NI,obj.NInII,obj.NInIInIII,obj.AV,obj.x,obj.X,obj.neighborsx,obj.neighborsX,obj.neighbors,obj.PD,obj.L,obj.Delta,obj.MatPars,obj.MatLaw);
        end

        function psi = energy(obj)
            psi = compute_energy(obj.Nr,obj.NI,obj.NInII,obj.NInIInIII,obj.AV,obj.x,obj.X,obj.neighborsx,obj.neighborsX,obj.neighbors,obj.PD,obj.L,obj.Delta,obj.MatPars,obj.MatLaw);
        end

        function avg_vols = avg_volumes(obj)
            avg_vols = compute_avg_volumes(obj.Nr,obj.NI,obj.NInII,obj.NInIInIII,obj.AV,obj.x,obj.X,obj.neighborsx,obj.neighborsX,obj.neighbors,obj.PD,obj.L,obj.Delta,obj.MatPars,obj.MatLaw);
        end

        function R = residual(obj)
            R = compute_residual(obj.Nr,obj.NI,obj.NInII,obj.NInIInIII,obj.AV,obj.x,obj.X,obj.neighborsx,obj.neighborsX,obj.neighbors,obj.PD,obj.L,obj.Delta,obj.MatPars,obj.MatLaw);
            % R = compute_residual_opt(obj.Nr,obj.NI,obj.NInII,obj.NInIInIII,obj.AV,obj.x,obj.X,obj.neighborsx,obj.neighborsX,obj.neighbors,obj.PD,obj.L,obj.Delta,obj.MatPars,obj.MatLaw);
        end

        function K = stiffness(obj)
            K = compute_stiffness(obj.Nr,obj.NI,obj.NInII,obj.NInIInIII,obj.AV,obj.x,obj.X,obj.neighborsx,obj.neighborsX,obj.neighbors,obj.PD,obj.L,obj.Delta,obj.MatPars,obj.MatLaw);
            % K = compute_stiffness_opt(obj.Nr,obj.NI,obj.NInII,obj.NInIInIII,obj.AV,obj.x,obj.X,obj.neighborsx,obj.neighborsX,obj.neighbors,obj.PD,obj.L,obj.Delta,obj.MatPars,obj.MatLaw);
        end

	end % end of methods

end % end of classdef

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vols = compute_volumes(Nr,NI,NInII,NInIInIII,AV,x,X,neighborsx,neighborsX,neighbors,PD,L,Delta,MatPars,MatLaw)

    vol1 = 0;
    vol2 = 0;
    vol3 = 0;

    NNgbr = length(neighbors);

    if ( PD==2 )

        A = AV;
        JI = A/NI;
        JInII = A^2/NInII;

    elseif ( PD==3 )

        V = AV;
        JI = V/NI;
        JInII = V^2/NInII;
        JInIInIII = V^3/NInIInIII;

    end

    if ( MatPars(1) ~= 0 )

        for i = 1 : NNgbr

            XiI = neighborsX(:,i) - X;
            xiI = neighborsx(:,i) - x;

            vol1tmp = JI;

            vol1 = vol1 + vol1tmp;

        end

    end

    if ( MatPars(2) ~= 0 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    XiI = neighborsX(:,i) - X;
                    xiI = neighborsx(:,i) - x;

                    XiII = neighborsX(:,j) - X;
                    xiII = neighborsx(:,j) - x;

                    vol2tmp = JInII;

                    vol2 = vol2 + vol2tmp;

                end

            end

        end

    end

    if ( MatPars(3) ~= 0 && PD==3 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    for k = 1 : NNgbr

                        if ( (k ~= i) && (k ~= j) )

                            XiI = neighborsX(:,i) - X;
                            xiI = neighborsx(:,i) - x;

                            XiII = neighborsX(:,j) - X;
                            xiII = neighborsx(:,j) - x;

                            XiIII = neighborsX(:,k) - X;
                            xiIII = neighborsx(:,k) - x;

                            vol3tmp = JInIInIII;

                            vol3 = vol3 + vol3tmp;

                        end

                    end

                end

            end

        end

    end

    vols = [ vol1 , vol2 , vol3 ];

end

function psi = compute_energy(Nr,NI,NInII,NInIInIII,AV,x,X,neighborsx,neighborsX,neighbors,PD,L,Delta,MatPars,MatLaw)

    psi = 0;
    psi1 = 0;
    psi2 = 0;
    psi3 = 0;

    NNgbr = length(neighbors);

    if ( PD==2 )

        A = AV;
        JI = A/NI;
        JInII = A^2/NInII;

    elseif ( PD==3 )

        V = AV;
        JI = V/NI;
        JInII = V^2/NInII;
        JInIInIII = V^3/NInIInIII;

    end

    if ( MatPars(1) ~= 0 )

        for i = 1 : NNgbr

            XiI = neighborsX(:,i) - X;
            xiI = neighborsx(:,i) - x;

            psi1tmp = JI * psifunc1(XiI,xiI,PD,L,Delta,MatPars,MatLaw);

            psi1 = psi1 + psi1tmp;

        end

    end

    if ( MatPars(2) ~= 0 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    XiI = neighborsX(:,i) - X;
                    xiI = neighborsx(:,i) - x;

                    XiII = neighborsX(:,j) - X;
                    xiII = neighborsx(:,j) - x;

                    psi2tmp = JInII * psifunc2(XiI,xiI,XiII,xiII,PD,L,Delta,MatPars,MatLaw);

                    psi2 = psi2 + psi2tmp;

                end

            end

        end

    end

    if ( MatPars(3) ~= 0 && PD==3 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    for k = 1 : NNgbr

                        if ( (k ~= i) && (k ~= j) )

                            XiI = neighborsX(:,i) - X;
                            xiI = neighborsx(:,i) - x;

                            XiII = neighborsX(:,j) - X;
                            xiII = neighborsx(:,j) - x;

                            XiIII = neighborsX(:,k) - X;
                            xiIII = neighborsx(:,k) - x;

                            psi3tmp = JInIInIII * psifunc3(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,L,Delta,MatPars,MatLaw);

                            psi3 = psi3 + psi3tmp;

                        end

                    end

                end

            end

        end

    end

    psi = psi1 + psi2 + psi3;

end

function avg_vols = compute_avg_volumes(Nr,NI,NInII,NInIInIII,AV,x,X,neighborsx,neighborsX,neighbors,PD,L,Delta,MatPars,MatLaw)

    L_avg = 0;
    A_avg = 0;
    V_avg = 0;

    l_avg = 0;
    a_avg = 0;
    v_avg = 0;

    NNgbr = length(neighbors);

    for i = 1 : NNgbr

        XiI = neighborsX(:,i) - X;
        xiI = neighborsx(:,i) - x;

        L_avg = L_avg + norm(XiI)/NI;
        l_avg = l_avg + norm(xiI)/NI;

    end

    for i = 1 : NNgbr

        for j = 1 : NNgbr

            if ( j ~= i )

                XiI = neighborsX(:,i) - X;
                xiI = neighborsx(:,i) - x;

                XiII = neighborsX(:,j) - X;
                xiII = neighborsx(:,j) - x;

                if ( PD==2 ) 

                    A = cross([XiI' 0] , [XiII' 0]);
                    a = cross([xiI' 0] , [xiII' 0]);

                elseif ( PD==3 ) 

                    A = cross(XiI',XiII');
                    a = cross(xiI',xiII');

                end

                tol = 1e-8;

                if ( norm(A) > tol && norm(XiI-XiII) < Delta )

                    A_avg = A_avg + norm(A)/NInII;
                    a_avg = a_avg + norm(a)/NInII;

                end

            end

        end

    end

    if ( PD==3 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    for k = 1 : NNgbr

                        if ( (k ~= i) && (k ~= j) )

                            XiI = neighborsX(:,i) - X;
                            xiI = neighborsx(:,i) - x;

                            XiII = neighborsX(:,j) - X;
                            xiII = neighborsx(:,j) - x;

                            XiIII = neighborsX(:,k) - X;
                            xiIII = neighborsx(:,k) - x;

                            V = XiI'*(cross(XiII',XiIII'))';
                            v = xiI'*(cross(xiII',xiIII'))';

                            tol = 1e-8;

                            if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

                                V_avg = V_avg + norm(V)/NInIInIII;
                                v_avg = v_avg + norm(v)/NInIInIII;

                            end

                        end

                    end

                end

            end

        end

    else

        V_avg = 0;
        v_avg = 0;
        
    end



    avg_vols = [ L_avg , A_avg , V_avg , l_avg , a_avg , v_avg ];

end

function R = compute_residual(Nr,NI,NInII,NInIInIII,AV,x,X,neighborsx,neighborsX,neighbors,PD,L,Delta,MatPars,MatLaw)

    R = zeros(PD,1);
    R1 = zeros(PD,1);
    R2 = zeros(PD,1);
    R3 = zeros(PD,1);
    ANG3 = zeros(PD,1);

    NNgbr = length(neighbors);

    if ( PD==2 )

        A = AV;
        JI = A/NI;
        JInII = A^2/NInII;

    elseif ( PD==3 )

        V = AV;
        JI = V/NI;
        JInII = V^2/NInII;
        JInIInIII = V^3/NInIInIII;

    end

    if ( MatPars(1) ~= 0 )

        for i = 1 : NNgbr

            XiI = neighborsX(:,i) - X;
            xiI = neighborsx(:,i) - x;

            R1tmp = JI * PP1(XiI,xiI,PD,L,Delta,MatPars,MatLaw);

            R1 = R1 + R1tmp;

        end

    end

    if ( MatPars(2) ~= 0 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    XiI = neighborsX(:,i) - X;
                    xiI = neighborsx(:,i) - x;

                    XiII = neighborsX(:,j) - X;
                    xiII = neighborsx(:,j) - x;

                    R2tmp = JInII * PP2(XiI,xiI,XiII,xiII,PD,L,Delta,MatPars,MatLaw);

                    R2 = R2 + R2tmp;

                end

            end

        end

    end

    if ( MatPars(3) ~= 0 && PD==3 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    for k = 1 : NNgbr

                        if ( (k ~= i) && (k ~= j) )

                            XiI = neighborsX(:,i) - X;
                            xiI = neighborsx(:,i) - x;

                            XiII = neighborsX(:,j) - X;
                            xiII = neighborsx(:,j) - x;

                            XiIII = neighborsX(:,k) - X;
                            xiIII = neighborsx(:,k) - x;

                            R3tmp = JInIInIII * PP3(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,L,Delta,MatPars,MatLaw);

                            R3 = R3 + R3tmp;

                        end

                    end

                end

            end

        end

    end

    R = R1 + R2 + R3;

end

function R = compute_residual_opt(Nr,NI,NInII,NInIInIII,AV,x,X,neighborsx,neighborsX,neighbors,PD,L,Delta,MatPars,MatLaw)

    R = zeros(PD,1);
    R1 = zeros(PD,1);
    R2 = zeros(PD,1);
    R3 = zeros(PD,1);
    ANG3 = zeros(PD,1);

    NNgbr = length(neighbors);

    if ( PD==2 )

        A = AV;
        JI = A/NI;
        JInII = A^2/NInII;

    elseif ( PD==3 )

        V = AV;
        JI = V/NI;
        JInII = V^2/NInII;
        JInIInIII = V^3/NInIInIII;

    end

    if ( MatPars(1) ~= 0 )

        for i = 1 : NNgbr

            XiI = neighborsX(:,i) - X;
            xiI = neighborsx(:,i) - x;

            R1tmp = JI * PP1(XiI,xiI,PD,L,Delta,MatPars,MatLaw);

            R1 = R1 + R1tmp;

        end

    end

    if ( MatPars(2) ~= 0 )

        for i = 1 : NNgbr

            for j = i+1 : NNgbr

                XiI = neighborsX(:,i) - X;
                xiI = neighborsx(:,i) - x;

                XiII = neighborsX(:,j) - X;
                xiII = neighborsx(:,j) - x;

                R2tmp = JInII * PP2opt(XiI,xiI,XiII,xiII,PD,L,Delta,MatPars,MatLaw);

                R2 = R2 + R2tmp;

            end

        end

    end

    if ( MatPars(3) ~= 0 && PD==3 )

        for i = 1 : NNgbr

            for j = (i+1) : NNgbr

                for k = (j+1) : NNgbr

                    XiI = neighborsX(:,i) - X;
                    xiI = neighborsx(:,i) - x;

                    XiII = neighborsX(:,j) - X;
                    xiII = neighborsx(:,j) - x;

                    XiIII = neighborsX(:,k) - X;
                    xiIII = neighborsx(:,k) - x;

                    R3tmp = JInIInIII * PP3opt(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,L,Delta,MatPars,MatLaw);

                    R3 = R3 + R3tmp;

                end

            end

        end

    end

    R = R1 + R2 + R3;

end

function K = compute_stiffness(Nr,NI,NInII,NInIInIII,AV,x,X,neighborsx,neighborsX,neighbors,PD,L,Delta,MatPars,MatLaw)

    NNgbr = length(neighbors);

    if ( PD==2 )

        A = AV;
        JI = A/NI;
        JInII = A^2/NInII;

    elseif ( PD==3 )

        V = AV;
        JI = V/NI;
        JInII = V^2/NInII;
        JInIInIII = V^3/NInIInIII;

    end

    NNgbrE = NNgbr+1; % neighbors including the point itself

    neighborsE = neighbors;
    neighborsEx = neighborsx;
    neighborsEX = neighborsX;

    neighborsE(NNgbrE) = Nr;
    neighborsEx(:,NNgbrE) = x;
    neighborsEX(:,NNgbrE) = X;

    K  = zeros(PD*PD,NNgbrE);
    K1 = zeros(PD*PD,NNgbrE);
    K2 = zeros(PD*PD,NNgbrE);
    K3 = zeros(PD*PD,NNgbrE);

    a = Nr;

    if ( MatPars(1) ~= 0 )

        for i = 1 : NNgbr

            XiI = neighborsX(:,i) - X;
            xiI = neighborsx(:,i) - x;

            AA1I = AA1(XiI,xiI,PD,L,Delta,MatPars,MatLaw);

            for b = 1 : NNgbrE

                K1tmp = JI * ( AA1I * ((neighbors(i)==neighborsE(b))-(a==neighborsE(b))) );

                K1(:,b) = K1(:,b) + K1tmp(:);

            end

        end

    end

    if ( MatPars(2) ~= 0 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    XiI = neighborsX(:,i) - X;
                    xiI = neighborsx(:,i) - x;

                    XiII = neighborsX(:,j) - X;
                    xiII = neighborsx(:,j) - x;

                    [AA2I , AA2J] = AA2(XiI,xiI,XiII,xiII,PD,L,Delta,MatPars,MatLaw);

                    for b = 1 : NNgbrE

                        K2tmp = JInII * ( AA2I * ((neighbors(i)==neighborsE(b))-(a==neighborsE(b)))...
                                        + AA2J * ((neighbors(j)==neighborsE(b))-(a==neighborsE(b))) );

                        K2(:,b) = K2(:,b) + K2tmp(:);

                    end

                end

            end

        end

    end

    if ( MatPars(3) ~= 0 && PD==3 )

        for i = 1 : NNgbr

            for j = 1 : NNgbr

                if ( j ~= i )

                    for k = 1 : NNgbr

                        if ( (k ~= i) && (k ~= j) )

                            XiI = neighborsX(:,i) - X;
                            xiI = neighborsx(:,i) - x;

                            XiII = neighborsX(:,j) - X;
                            xiII = neighborsx(:,j) - x;

                            XiIII = neighborsX(:,k) - X;
                            xiIII = neighborsx(:,k) - x;

                            [AA3I , AA3J , AA3K] = AA3(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,L,Delta,MatPars,MatLaw);

                            for b = 1 : NNgbrE

                                K3tmp = JInIInIII * ( AA3I * ((neighbors(i)==neighborsE(b))-(a==neighborsE(b)))...
                                                      + AA3J * ((neighbors(j)==neighborsE(b))-(a==neighborsE(b)))...
                                                      + AA3K * ((neighbors(k)==neighborsE(b))-(a==neighborsE(b))) );

                                K3(:,b) = K3(:,b) + K3tmp(:);

                            end

                        end

                    end

                end

            end

        end

    end

    K = K1 + K2 + K3;

end

function K = compute_stiffness_opt(Nr,NI,NInII,NInIInIII,AV,x,X,neighborsx,neighborsX,neighbors,PD,L,Delta,MatPars,MatLaw)

    NNgbr = length(neighbors);

    % if ( PD==2 )

    %     A = AV;
    %     J = A/NNgbr;

    % elseif ( PD==3 )

    %     V = AV;
    %     J = V/NNgbr;

    % end

    if ( PD==2 )

        A = AV;
        JI = A/NI;
        JInII = A^2/NInII;

    elseif ( PD==3 )

        V = AV;
        JI = V/NI;
        JInII = V^2/NInII;
        JInIInIII = V^3/NInIInIII;

    end

    NNgbrE = NNgbr+1; % neighbors including the point itself

    neighborsE = neighbors;
    neighborsEx = neighborsx;
    neighborsEX = neighborsX;

    neighborsE(NNgbrE) = Nr;
    neighborsEx(:,NNgbrE) = x;
    neighborsEX(:,NNgbrE) = X;

    K = zeros(PD*PD,NNgbrE);
    K1 = zeros(PD*PD,NNgbrE);
    K2 = zeros(PD*PD,NNgbrE);
    K3 = zeros(PD*PD,NNgbrE);

    a = Nr;

    if ( MatPars(1) ~= 0 )

        for i = 1 : NNgbr

            XiI = neighborsX(:,i) - X;
            xiI = neighborsx(:,i) - x;

            AA1I = AA1(XiI,xiI,PD,L,Delta,MatPars,MatLaw);

            for b = 1 : NNgbrE

                K1tmp = JI * ( AA1I * ((neighbors(i)==neighborsE(b))-(a==neighborsE(b))) );

                K1(:,b) = K1(:,b) + K1tmp(:);

            end

        end

    end

    if ( MatPars(2) ~= 0 )

        for i = 1 : NNgbr

            for j = (i+1) : NNgbr

                XiI = neighborsX(:,i) - X;
                xiI = neighborsx(:,i) - x;

                XiII = neighborsX(:,j) - X;
                xiII = neighborsx(:,j) - x;

                [AA2I , AA2J] = AA2opt(XiI,xiI,XiII,xiII,PD,L,Delta,MatPars,MatLaw);

                for b = 1 : NNgbrE

                    K2tmp = JInII * ( AA2I * ((neighbors(i)==neighborsE(b))-(a==neighborsE(b)))...
                                    + AA2J * ((neighbors(j)==neighborsE(b))-(a==neighborsE(b))) );

                    K2(:,b) = K2(:,b) + K2tmp(:);

                end

            end

        end

    end

    if ( MatPars(3) ~= 0 && PD==3 )

        for i = 1 : NNgbr

            for j = (i+1) : NNgbr

                for k = (j+1) : NNgbr

                    XiI = neighborsX(:,i) - X;
                    xiI = neighborsx(:,i) - x;

                    XiII = neighborsX(:,j) - X;
                    xiII = neighborsx(:,j) - x;

                    XiIII = neighborsX(:,k) - X;
                    xiIII = neighborsx(:,k) - x;

                    [AA3I , AA3J , AA3K] = AA3opt(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,L,Delta,MatPars,MatLaw);

                    for b = 1 : NNgbrE

                        K3tmp = JInIInIII * ( AA3I * ((neighbors(i)==neighborsE(b))-(a==neighborsE(b)))...
                                              + AA3J * ((neighbors(j)==neighborsE(b))-(a==neighborsE(b)))...
                                              + AA3K * ((neighbors(k)==neighborsE(b))-(a==neighborsE(b))) );

                        K3(:,b) = K3(:,b) + K3tmp(:);

                    end

                end

            end

        end

    end

    K = K1 + K2 + K3;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%       ONE-NEIGHBOR ENERGY        %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = psifunc1(XiI,xiI,PD,Lo,Delta,MatPars,MatLaw)

    l = norm(xiI);
    L = norm(XiI);

    s = (l-L)/L;

    CC = MatPars(1);

    out = 1/2 * CC * L * s^2;

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        out = 1/L * out;

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%       TWO-NEIGHBOR ENERGY        %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = psifunc2(XiI,xiI,XiII,xiII,PD,Lo,Delta,MatPars,MatLaw)

    out = 0;

    if ( PD==2 ) 

        A = cross([XiI' 0] , [XiII' 0]);
        a = cross([xiI' 0] , [xiII' 0]);

    elseif ( PD==3 ) 

        A = cross(XiI',XiII');
        a = cross(xiI',xiII');

    end

    tol = 1e-8;

    if ( norm(A) > tol && norm(XiI-XiII) < Delta )

        s = (norm(a)-norm(A))/norm(A);

        CC = MatPars(2);

        out = 1/2 * CC * norm(A) * s^2;

    end

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        if ( norm(A) > tol && norm(XiI-XiII) < Delta )

            out = 1/(norm(A)) * out;

        end

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%      THREE-NEIGHBOR ENERGY       %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = psifunc3(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,Lo,Delta,MatPars,MatLaw)

    out = 0;

    V = XiI'*(cross(XiII',XiIII'))';
    v = xiI'*(cross(xiII',xiIII'))';

    tol = 1e-8;

    if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

        s = (norm(v)-norm(V))/norm(V);

        CC = MatPars(3);

        out = 1/2 * CC * norm(V) * s^2;

    end

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        if ( norm(V) > tol )

            out = 1/(norm(V)) * out;

        end

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%       ONE-NEIGHBOR FORCE        %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = PP1(XiI,xiI,PD,Lo,Delta,MatPars,MatLaw)

    l = norm(xiI);
    L = norm(XiI);

    s = (l-L)/L;

    eta = xiI/l;

    CC = MatPars(1);

    out = CC * eta * s;

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        out = 1/L * out;

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%       TWO-NEIGHBOR FORCE        %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = PP2(XiI,xiI,XiII,xiII,PD,Lo,Delta,MatPars,MatLaw)

    out = zeros(PD,1);

    if ( PD==2 ) 

        A = cross([XiI' 0] , [XiII' 0]);
        a = cross([xiI' 0] , [xiII' 0]);

    elseif ( PD==3 ) 

        A = cross(XiI',XiII');
        a = cross(xiI',xiII');

    end

    tol = 1e-8;

    if ( norm(A) > tol && norm(XiI-XiII) < Delta )

        CC = MatPars(2);

        G = 1/norm(A) - 1/norm(a);

        H = (xiII'*xiII) * xiI - (xiII'*xiI) * xiII;

        out = 2 * CC * G * H;

    end

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        if ( norm(A) > tol )

            out = 1/(norm(A)) * out;

        end

    end

end

function out = PP2opt(XiI,xiI,XiII,xiII,PD,Lo,Delta,MatPars,MatLaw)

    out = zeros(PD,1);

    if ( PD==2 ) 

        A = cross([XiI' 0] , [XiII' 0]);
        a = cross([xiI' 0] , [xiII' 0]);

    elseif ( PD==3 ) 

        A = cross(XiI',XiII');
        a = cross(xiI',xiII');

    end

    tol = 1e-8;

    if ( norm(A) > tol && norm(XiI-XiII) < Delta )

        CC = MatPars(2);

        G = 1/norm(A) - 1/norm(a);

        Ha = (xiII'*xiII) * xiI - (xiII'*xiI) * xiII;
        Hb = (xiI'*xiI) * xiII - (xiI'*xiII) * xiI;

        out = 2 * CC * G * (Ha+Hb);

    end

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        if ( norm(A) > tol )

            out = 1/(norm(A)) * out;

        end

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%       THREE-NEIGHBOR FORCE        %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = PP3(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,Lo,Delta,MatPars,MatLaw)

    out = zeros(PD,1);

    V = XiI'*(cross(XiII',XiIII'))';
    v = xiI'*(cross(xiII',xiIII'))';

    tol = 1e-8;

    if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

        CC = MatPars(3);

        out = 3 * CC * cross(xiII',xiIII')' * (1/abs(V)-1/abs(v)) * v;

    end

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        if ( abs(V) > tol )

            out = 1/(norm(V)) * out;

        end

    end

end

function out = PP3opt(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,Lo,Delta,MatPars,MatLaw)

    out = zeros(PD,1);

    V = XiI'*(cross(XiII',XiIII'))';
    v = xiI'*(cross(xiII',xiIII'))';

    tol = 1e-8;

    if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

        CC = MatPars(3);

        Ha = cross(xiII',xiIII')';
        Hb = cross(xiIII',xiI')';
        Hc = cross(xiI',xiII')';

        Hd = -Ha;
        He = -Hb;
        Hf = -Hc;

        out = 3 * CC * ((Ha+Hb+Hc)-(Hd+He+Hf)) * (1/abs(V)-1/abs(v)) * v;

    end

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        if ( abs(V) > tol )

            out = 1/(norm(V)) * out;

        end

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%       THREE-NEIGHBOR ANG. MOM.        %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = ANGM3(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,Lo,Delta,MatPars,MatLaw)

    out = zeros(PD,1);

    V = XiI'*(cross(XiII',XiIII'))';
    v = xiI'*(cross(xiII',xiIII'))';

    tol = 1e-8;

    if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

        CC = MatPars(3);

        out = 3 * CC * abs(v-V)/abs(V) * v/abs(v) * ( cross(xiI',(cross(xiII',xiIII'))) )';

    end

end

function out = ANGM3b(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,Lo,Delta,MatPars,MatLaw)

    out = zeros(PD,1);

    V = XiI'*(cross(XiII',XiIII'))';
    v = xiI'*(cross(xiII',xiIII'))';

    tol = 1e-8;

    if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

        CC = MatPars(3);

        out = 3 * CC * (v-V)/V * ( cross(xiI',(cross(xiII',xiIII'))) )';

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%       ONE-NEIGHBOR STIFFNESS        %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = AA1(XiI,xiI,PD,Lo,Delta,MatPars,MatLaw)

    l = norm(xiI);
    L = norm(XiI);

    II = eye(PD);

    s = (l-L)/L;

    eta = xiI/l;

    eta_dyad_eta = eta * eta';

    CC = MatPars(1);

    out = CC * ( s/l * ( II - eta_dyad_eta ) + 1/L * eta_dyad_eta ); 

    switch MatLaw

    case 'SS'

        out = out;

    case 'AJ'

        out = 1/L * out;

    end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%       TWO-NEIGHBOR STIFFNESS        %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ outI , outJ ] = AA2(XiI,xiI,XiII,xiII,PD,Lo,Delta,MatPars,MatLaw)

    outI = zeros(PD);
    outJ = zeros(PD);

    if ( PD==2 ) 

        A = cross([XiI' 0] , [XiII' 0]);
        a = cross([xiI' 0] , [xiII' 0]);

    elseif ( PD==3 ) 

        A = cross(XiI',XiII');
        a = cross(xiI',xiII');

    end

    AA = norm(A);
    aa = norm(a);

    tol = 1e-8;

    if ( AA > tol && norm(XiI-XiII) < Delta )

        II = eye(PD);

        CC = MatPars(2);

        BBI1 = xiII'*xiII * II - xiII*xiII';

        BBJ1 = 2*xiI*xiII' - xiI'*xiII * II - xiII*xiI';

        eInII = (xiII'*xiII)*xiI - (xiI'*xiII)*xiII;
        eIInI = (xiI'*xiI)*xiII - (xiII'*xiI)*xiI;

        outI = 2 * CC * (1/AA-1/aa) * BBI1 + 2 * CC * 1/(aa^3) * (eInII*eInII');
        outJ = 2 * CC * (1/AA-1/aa) * BBJ1 + 2 * CC * 1/(aa^3) * (eInII*eIInI');

    end

    switch MatLaw

    case 'SS'

        outI = outI;
        outJ = outJ;

    case 'AJ'

        if ( AA > tol )

            outI = 1/AA * outI;
            outJ = 1/AA * outJ;

        end

    end

end

function [ outI , outJ ] = AA2opt(XiI,xiI,XiII,xiII,PD,Lo,Delta,MatPars,MatLaw)

    outI = zeros(PD);
    outJ = zeros(PD);

    if ( PD==2 ) 

        A = cross([XiI' 0] , [XiII' 0]);
        a = cross([xiI' 0] , [xiII' 0]);

    elseif ( PD==3 ) 

        A = cross(XiI',XiII');
        a = cross(xiI',xiII');

    end

    tol = 1e-8;

    if ( norm(A) > tol && norm(XiI-XiII) < Delta )

        II = eye(PD);

        CC = MatPars(2);

        BBI1a = xiII'*xiII * II - xiII*xiII';

        BBI2a = (xiII'*xiII) * (xiII'*xiII) * (xiI*xiI')...
           - (xiII'*xiII) * (xiI'*xiII) * (xiI*xiII')...
           - (xiII'*xiII) * (xiI'*xiII) * (xiII*xiI')...
           + (xiI'*xiII) * (xiI'*xiII) * (xiII*xiII');

        BBJ1a = 2*xiI*xiII' - xiI'*xiII * II - xiII*xiI';

        BBJ2a = (xiII'*xiII) * (xiI'*xiII) * (xiI*xiI')...
           - (xiII'*xiII) * (xiI'*xiI) * (xiI*xiII')...
           - (xiI'*xiII) * (xiI'*xiII) * (xiII*xiI')...
           + (xiII'*xiI) * (xiI'*xiI) * (xiII*xiII');

        BBI1b = xiI'*xiI * II - xiI*xiI';

        BBI2b = (xiI'*xiI) * (xiI'*xiI) * (xiII*xiII')...
           - (xiI'*xiI) * (xiII'*xiI) * (xiII*xiI')...
           - (xiI'*xiI) * (xiII'*xiI) * (xiI*xiII')...
           + (xiII'*xiI) * (xiII'*xiI) * (xiI*xiI');

        BBJ1b = 2*xiII*xiI' - xiII'*xiI * II - xiI*xiII';

        BBJ2b = (xiI'*xiI) * (xiII'*xiI) * (xiII*xiII')...
           - (xiI'*xiI) * (xiII'*xiII) * (xiII*xiI')...
           - (xiII'*xiI) * (xiII'*xiI) * (xiI*xiII')...
           + (xiI'*xiII) * (xiII'*xiII) * (xiI*xiI');

        outI = 2 * CC * (1/(norm(A))-1/(norm(a))) * (BBI1a + BBJ1b) + 2 * CC * 1/(norm(a))^3 * (BBI2a - BBJ2b);

        outJ = 2 * CC * (1/(norm(A))-1/(norm(a))) * (BBJ1a + BBI1b) + 2 * CC * 1/(norm(a))^3 * (BBI2b - BBJ2a);

    end

    switch MatLaw

    case 'SS'

        outI = outI;
        outJ = outJ;

    case 'AJ'

        if ( norm(A) > tol )

            outI = 1/(norm(A)) * outI;
            outJ = 1/(norm(A)) * outJ;

        end

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%       THREE-NEIGHBOR STIFFNESS        %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ outI , outJ , outK ] = AA3(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,Lo,Delta,MatPars,MatLaw)

    outI = zeros(PD);
    outJ = zeros(PD);
    outK = zeros(PD);

    V = XiI'*(cross(XiII',XiIII'))';
    v = xiI'*(cross(xiII',xiIII'))';

    tol = 1e-8;

    if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

        II = eye(PD);

        CC = MatPars(3);

        BBI1 = (cross(xiII',xiIII'))' * (cross(xiII',xiIII'));

        BBJ1 = [ 0 , xiIII(3) , -xiIII(2) ; -xiIII(3) 0 xiIII(1) ; xiIII(2) -xiIII(1) 0 ];

        BBJ2 = (cross(xiII',xiIII'))' * (cross(xiIII',xiI'));

        BBK1 = [ 0 , xiII(3) , -xiII(2) ; -xiII(3) 0 xiII(1) ; xiII(2) -xiII(1) 0 ];

        BBK2 = (cross(xiII',xiIII'))' * (cross(xiI',xiII'));

        outI = 3 * CC * ( 1/(abs(V)) ) * BBI1 ;
        outJ = 3 * CC * ( 1/(abs(V))-1/(abs(v)) ) * v * BBJ1 + 3 * CC * ( 1/(abs(V)) ) * BBJ2;
        outK =-3 * CC * ( 1/(abs(V))-1/(abs(v)) ) * v * BBK1 + 3 * CC * ( 1/(abs(V)) ) * BBK2;

    end

    switch MatLaw

    case 'SS'

        outI = outI;
        outJ = outJ;
        outK = outK;

    case 'AJ'

        if ( norm(V) > tol )

            outI = 1/(norm(V)) * outI;
            outJ = 1/(norm(V)) * outJ;
            outK = 1/(norm(V)) * outK;

        end

    end

end

function [ outI , outJ , outK ] = AA3opt(XiI,xiI,XiII,xiII,XiIII,xiIII,PD,Lo,Delta,MatPars,MatLaw)

    outI = zeros(PD);
    outJ = zeros(PD);
    outK = zeros(PD);

    V = XiI'*(cross(XiII',XiIII'))';
    v = xiI'*(cross(xiII',xiIII'))';

    tol = 1e-8;

    if ( abs(V) > tol && norm(XiI-XiII) < Delta && norm(XiI-XiIII) < Delta && norm(XiII-XiIII) < Delta )

        II = eye(PD);

        CC = MatPars(3);

        BBI1a = (cross(xiII',xiIII'))' * (cross(xiII',xiIII'));
        BBJ1a = [ 0 , xiIII(3) , -xiIII(2) ; -xiIII(3) 0 xiIII(1) ; xiIII(2) -xiIII(1) 0 ];
        BBJ2a = (cross(xiII',xiIII'))' * (cross(xiIII',xiI'));
        BBK1a = [ 0 , xiII(3) , -xiII(2) ; -xiII(3) 0 xiII(1) ; xiII(2) -xiII(1) 0 ];
        BBK2a = (cross(xiII',xiIII'))' * (cross(xiI',xiII'));

        BBI1b = (cross(xiIII',xiI'))' * (cross(xiIII',xiI'));
        BBJ1b = [ 0 , xiI(3) , -xiI(2) ; -xiI(3) 0 xiI(1) ; xiI(2) -xiI(1) 0 ];
        BBJ2b = (cross(xiIII',xiI'))' * (cross(xiI',xiII'));
        BBK1b = [ 0 , xiIII(3) , -xiIII(2) ; -xiIII(3) 0 xiIII(1) ; xiIII(2) -xiIII(1) 0 ];
        BBK2b = (cross(xiIII',xiI'))' * (cross(xiII',xiIII'));

        BBI1c = (cross(xiI',xiII'))' * (cross(xiI',xiII'));
        BBJ1c = [ 0 , xiII(3) , -xiII(2) ; -xiII(3) 0 xiII(1) ; xiII(2) -xiII(1) 0 ];
        BBJ2c = (cross(xiI',xiII'))' * (cross(xiII',xiIII'));
        BBK1c = [ 0 , xiI(3) , -xiI(2) ; -xiI(3) 0 xiI(1) ; xiI(2) -xiI(1) 0 ];
        BBK2c = (cross(xiI',xiII'))' * (cross(xiIII',xiI'));

        BBI1d = (cross(xiIII',xiII'))' * (cross(xiIII',xiII'));
        BBJ1d = [ 0 , xiII(3) , -xiII(2) ; -xiII(3) 0 xiII(1) ; xiII(2) -xiII(1) 0 ];
        BBJ2d = (cross(xiIII',xiII'))' * (cross(xiII',xiI'));
        BBK1d = [ 0 , xiIII(3) , -xiIII(2) ; -xiIII(3) 0 xiIII(1) ; xiIII(2) -xiIII(1) 0 ];
        BBK2d = (cross(xiIII',xiII'))' * (cross(xiI',xiIII'));

        BBI1e = (cross(xiI',xiIII'))' * (cross(xiI',xiIII'));
        BBJ1e = [ 0 , xiIII(3) , -xiIII(2) ; -xiIII(3) 0 xiIII(1) ; xiIII(2) -xiIII(1) 0 ];
        BBJ2e = (cross(xiI',xiIII'))' * (cross(xiIII',xiII'));
        BBK1e = [ 0 , xiI(3) , -xiI(2) ; -xiI(3) 0 xiI(1) ; xiI(2) -xiI(1) 0 ];
        BBK2e = (cross(xiI',xiIII'))' * (cross(xiII',xiI'));

        BBI1f = (cross(xiII',xiI'))' * (cross(xiII',xiI'));
        BBJ1f = [ 0 , xiI(3) , -xiI(2) ; -xiI(3) 0 xiI(1) ; xiI(2) -xiI(1) 0 ];
        BBJ2f = (cross(xiII',xiI'))' * (cross(xiI',xiIII'));
        BBK1f = [ 0 , xiII(3) , -xiII(2) ; -xiII(3) 0 xiII(1) ; xiII(2) -xiII(1) 0 ];
        BBK2f = (cross(xiII',xiI'))' * (cross(xiIII',xiII'));

        CCV = 3 * CC * ( 1/(abs(V)) );
        CCVV = 3 * CC * ( 1/(abs(V))-1/(abs(v)) );

        outI = CCV * ( BBI1a + BBK2b + BBJ2c + BBI1d + BBJ2e + BBK2f ) + CCVV * v * ( - BBK1b + BBJ1c - BBJ1e + BBK1f );
        outJ = CCV * ( BBJ2a + BBI1b + BBK2c + BBK2d + BBI1e + BBJ2f ) + CCVV * v * ( BBJ1a - BBK1c + BBK1d - BBJ1f ); 
        outK = CCV * ( BBK2a + BBJ2b + BBI1c + BBJ2d + BBK2e + BBI1f ) + CCVV * v * (  -BBK1a + BBJ1b - BBJ1d + BBK1e ); 

    end

    switch MatLaw

    case 'SS'

        outI = outI;
        outJ = outJ;
        outK = outK;

    case 'AJ'

        if ( norm(V) > tol )

            outI = 1/(norm(V)) * outI;
            outJ = 1/(norm(V)) * outJ;
            outK = 1/(norm(V)) * outK;

        end

    end

end
