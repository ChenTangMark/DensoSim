
###################
# agent.py
# this file contains classes to define an agent,
# each agent generates control input to sent to the vehicle
###################
from __future__ import division
from direct.stdpy.file import *
import math
import numpy as np
from cvxopt import matrix, solvers
import utility
import optimization as opt
import copy
solvers.options['feastol'] = 1e-10

# this is a basic agent structure that generate zero input
class basicAgent:
    def __init__(self):
        return

    def setVehicle(self,vehicle):
        self.vehicle=vehicle

    # Get global position (x,y)
    def getPos(self):
        return self.vehicle.sensor.getSelfPos()

    # Get current velocity v
    def getVelocity(self):
        return self.vehicle.sensor.getVelocity()

    def getVelocityVector(self):
        return self.vehicle.getVelocityVector()

    # Get direction of the vehilce in vector form
    def getDirection(self):
        return self.vehicle.sensor.getSelfAngle()

    # Get the state of the vehicle: x,y,v,theta
    def getState(self):
        state = self.getPos()
        state.append(self.getVelocity())
        state.append(utility.vec2ang(self.getDirection()))
        return state

    # Get distance to the lane center (lane ID=lf)
    def getDis(self,lf=1):
        return self.vehicle.sensor.getCordPos(lf)[0]

    # Get the angle between the vehicle and the road
    def getAngle(self):
        return self.vehicle.sensor.getCordAngle()

    # Get angular velocity of the vehicle theta dot
    def getAngularVelocity(self):
        return self.vehicle.getAngleVelocity()

    def doControl(self):
        return [0,0,0]

# the laneKeepingAgent keeps driving in the centerline
class laneKeepingAgent(basicAgent):
    def __init__(self,vGain=20,thetaGain=20,desiredV=35,laneId=0):
        self.vGain = vGain
        self.thetaGain = thetaGain
        self.desiredV = desiredV
        self.targetLane = laneId

    def setTargetLane(self,laneId):
        self.targetLane = laneId

    # Feedback control is based on the angle difference and position difference
    def getFeedbackControl(self,diffAngle,diffPos,diffPosV):
        
        acceleration=self.vGain*(self.desiredV*math.cos(self.getAngle())-self.getVelocity())
        steer=-self.thetaGain*diffAngle/(self.getVelocity()+1)-3*self.vehicle.getAngleVelocity()-5*diffPos-5*diffPosV
        return [acceleration,steer]

    def doControl(self):
        diffPosV=self.vehicle.sensor.getCordVelocity((self.vehicle.sensor.getLineInRange(0,2,self.targetLane)))
        fb=self.getFeedbackControl(self.getAngle(),self.getDis(self.targetLane),diffPosV)
        return [fb[0],fb[1],0]

# the previewAgent adds preview control to the laneKeepingAgent
class previewAgent(laneKeepingAgent):
    def __init__(self,vGain=5000,thetaGain=20,desiredV=15,laneId=0,ffGain=1000):
        self.vGain = vGain
        self.thetaGain = thetaGain
        self.ffGain=ffGain
        self.desiredV = desiredV
        self.targetLane = laneId
        self.ts = 1.0/15

    def getPreview(self,laneId=0,length=20):
        return self.vehicle.sensor.getLineInRange(0,length,laneId)

    # Feedforward control based on future trajectory
    def getFeedforwardControl(self,futureTraj):
        # Steering
        vehicleDirection = self.getDirection()
        displacement = []

        for index in range(len(futureTraj)-1):
            trajDirection = [futureTraj[index+1][0]-futureTraj[0][0],futureTraj[index+1][1]-futureTraj[0][1]]
            vector = utility.perpLength(trajDirection,vehicleDirection)
            length = np.linalg.norm(vector)
            #displacement.append((-trajDirection[0]*vehicleDirection[1]+trajDirection[1]*vehicleDirection[0])/(vehicleDirection[0]**2+vehicleDirection[1]**2)*vector)
            displacement.append(length*vector)

        steerV = 0
        for index in range(len(displacement)):
            steerV += displacement[index]/(index + 1)/(self.getVelocity()+1)
        steerV =  steerV / (index + 1)


        # Accelaration
        vehicleVelocity = self.getVelocity()
        deltav = []

        for index in range(len(futureTraj)-1):
            refVel = np.linalg.norm(np.array([futureTraj[index+1][0]-futureTraj[index][0],futureTraj[index+1][1]-futureTraj[index][1]]))
            deltav.append(refVel - vehicleVelocity)

        accelaration = 0
        for index in range(len(displacement)):
            accelaration += deltav[index]/(index + 1)
        accelaration = accelaration / (index+1)
        #self.desiredV = np.linalg.norm(np.array([(futureTraj[-1][0]-futureTraj[0][0]),(futureTraj[-1][1]-futureTraj[0][1])]))/index/self.ts

        return [accelaration,steerV]

    # Preview control for lane id = lf
    def previewController(self,laneId=0):
        futureTraj = self.getPreview(laneId)
        diffPosV=self.vehicle.sensor.getCordVelocity(futureTraj)
        ff = self.getFeedforwardControl(futureTraj)
        fb = self.getFeedbackControl(self.getAngle(),self.getDis(laneId),diffPosV)
        ff=ff*self.ffGain
        acceleration = ff[0]+fb[0]
        steerV = ff[1]+fb[1]
        return [acceleration,steerV,0]

    def doControl(self):
        return self.previewController(self.targetLane)

# the autoBrakeAgent brakes when encontering obstacles in front
class autoBrakeAgent(previewAgent):
    def __init__(self,vGain=50,thetaGain=20,desiredV=15,headway=20):
        self.vGain=vGain
        self.thetaGain=thetaGain
        self.desiredV=desiredV
        self.safeHeadway = headway

    # Returns relative position vector and relative velocity vector
    def getSurroundVehicleRelateState(self,num):
        return self.vehicle.sensor.getSurroundVehicleRelateState(num)

    # Returns position vector and velocity vector
    def getSurroundVehicleState(self,num):
        return self.vehicle.sensor.getSurroundVehicleState(num)

    def getHeadway(self):
        vehicleDirection = self.vehicle.sensor.getSelfAngle()
        [relateX,relateV]=self.getSurroundVehicleRelateState(1)
        if np.linalg.norm(np.cross(vehicleDirection,relateX))<2:
            dis = np.dot(vehicleDirection,relateX)
            if dis<0:
                return self.safeHeadway
            else:
                return dis
        else:
            return self.safeHeadway

    def autoBrake(self,laneId=0):
        headway = self.getHeadway()
        if headway < self.safeHeadway:

            if headway < self.safeHeadway/2:
                #print('brake')
                return [0,0,100/(headway+0.1)]
            else:
                return [-50/(headway+0.1),self.previewController(laneId)[1]/(self.safeHeadway/headway)**0.5,0]
        else:
            return self.previewController(laneId)

    def doControl(self,laneId=0):
        return self.autoBrake(laleId)

# the 
class laneChangeAgent(autoBrakeAgent):
    def __init__(self,vGain=50,thetaGain=20,desiredV=25,laneId=0,num=1,radiu=500,headway=20):
        self.vGain=vGain
        self.thetaGain=thetaGain
        self.desiredV=desiredV
        self.safeHeadway = headway
        self.traj = [[0,0]]
        self.numSurrounding = num # number of surrounding vehicles
        self.h=2
        self.dmin = 10
        self.ts = 1.00/15
        self.targetLane=laneId
        self.changeFlag=0
        self.phiFlag=0
        self.frontObs=0
        self.radiu=radiu
        self.Range=15
        self.alpha=50
        self.ita=2
        self.changeThres=1.3
        self.previousInput=[0,0]

    # The Efficiency controller
    def getTrajectory(self):
        #return self.getPreview(1,25)

        horizon = 20
        refTraj = []
        if len(self.traj)>1:
            refTraj = self.traj[:]

        preTraj = self.getPreview(self.getCurrLaneId(),horizon)
        #print(self.getCurrLaneId())
        if len(self.traj)>1:
            i = len(self.traj)
        else:
            i = len(self.traj)-1

        while i < horizon:
            i = i+1
            refTraj.append(preTraj[i])
        #print(len(refTraj))

        x0 = self.getPos()

        # get trajectory of surrounding vehicles
        obs = []
        for i in range(self.numSurrounding):
            obs.append(self.getPrediction(i,horizon))
        #print(obs)
        traj = opt.CFS_FirstOrder(x0,refTraj,obs,horizon,self.ts)
        return traj


    # Get lateral distance to a trajectory; return (dis, angle, index)
    def getDisAngle(self,x0,theta,traj):
        l1 = np.linalg.norm(utility.substruct(traj[0],x0))
        for index in range(len(traj)-1):
            l2 = np.linalg.norm(utility.substruct(traj[index+1],x0))
            direction = utility.substruct(traj[index+1],traj[index])
            angle = utility.vec2ang(direction)

            if l1**2+l2**2 < np.linalg.norm(direction)**2 or l1 < l2:
                return (utility.perpLength(utility.substruct(x0,traj[index]),direction),(theta-angle+math.pi) % (2*math.pi)-math.pi,index)


    # Overwrite the preview controller in Preview Agent
    def previewController(self):

        '''h = self.h

        if len(self.traj)<h:
            self.traj = list(self.getTrajectory())'''

        self.traj = self.getPreview(self.targetLane,25)
        futureTraj = self.traj
        #print(futureTraj)

        ff = [0,0]#self.getFeedforwardControl(futureTraj)
        
        #dis, angle, index = self.getDisAngle(self.getPos(),utility.vec2ang(self.getDirection()),futureTraj)
        diffPosV=self.vehicle.sensor.getCordVelocity(futureTraj)
        fb = self.getFeedbackControl(self.getAngle(),self.getDis(self.targetLane),diffPosV)
        #print(dis-self.getDis(),angle-self.getAngle())

        '''i = 0
        while i <= index:
            self.traj.pop(0)
            i += 1'''

        acceleration = ff[0]+fb[0]
        steerV = ff[1]+fb[1]

        steeringLimit=45

        if steerV>steeringLimit:
          steerV=steeringLimit
        if steerV<-steeringLimit:
          steerV=-steeringLimit

        return [acceleration,steerV,0]

    def getCurrLaneId(self):
        dev=-self.vehicle.sensor.getCordPos(0)[0]-6
        if dev<-4:
            return 0
        if dev<0:
            return 1
        if dev<4:
            return 2
        return 3 

    def getSurrVehicleLaneId(self,num):
        dev=-self.vehicle.sensor.getSurroundVehicle(num).sensor.getCordPos(0)[0]-6
        if dev<-4:
            return 0
        if dev<0:
            return 1
        if dev<4:
            return 2
        return 3 


    def getLaneDirection(self):
        futureTraj = self.getPreview(1,2)
        laneDirection = np.array([futureTraj[1][0]-futureTraj[0][0],futureTraj[1][1]-futureTraj[0][1]])
        laneDirection = laneDirection/np.linalg.norm(laneDirection)
        return laneDirection

    # Prediction of the future trajectory of surrounding vehicles
    # Here we use constant speed model
    def getPrediction(self,id,horizon):
        [x0,v0] = self.getSurroundVehicleState(id)

        traj = np.zeros((horizon, 2))
        for i in range(horizon):
            traj[i] = x0+v0*self.ts*(i+1)*0.1
        #print(traj)
        return traj

    def changeTargetLane(self):
        self.targetLane=self.getCurrLaneId()

    def ACC(self):
        currLaneObsId=[]
        for index in range(self.numSurrounding):
            if self.getSurrVehicleLaneId(index+1)==self.targetLane:#self.getCurrLaneId():
                currLaneObsId.append(index)
        d=1000
	obs=-1
        for index in currLaneObsId:
            [relateX,relateV]=self.getSurroundVehicleRelateState(index+1)
            if relateX[1]<d and relateX[1]>0:
                d=relateX[1]
                obs=index
	if obs!=-1:
		# Fixed Initials
		ACCHead=1
		Ts=0.1
		N=50
		dim=1
		vr=35
		Vdiff = -np.eye(N*dim)+np.diag(np.ones((N-1)*dim),k=dim)
		Adiff = -Vdiff-np.diag(np.ones((N-1)*dim),k=dim)+np.diag(np.ones(N-2),k=dim*2)
		V=Vdiff[0:(N-1)*dim,:]/Ts
		A=Adiff[0:(N-2)*dim,:]/(Ts*Ts)
		a1=2;a2=0.1;a3=1
		QV=a1*np.eye(N-1)
		QA=a2*np.eye(N-2)
		Qobs=a3*np.eye(N)
		vl=5
		vh=30
		al=-5
		ah=2
		Ae=np.zeros((2,N))
		be=np.zeros((2,1))
		Ae[0,0]=1
		Ae[1,0:2]=np.array([-1,1])/Ts
		Ah=np.zeros((2*(N-1)+2*(N-2),N))
		bh=np.zeros((2*(N-1)+2*(N-2),1))
		Ah[0:N-1,:]=V
		Ah[N-1:2*(N-1),:]=-V
		Ah[2*(N-1):2*(N-1)+N-2,:]=A
		Ah[3*(N-1)-1:2*(N-1)+2*(N-2),:]=-A
		bh[0:N-1]=vh*np.ones((N-1,1))
		bh[N-1:2*(N-1)]=-vl*np.ones((N-1,1))
		bh[2*(N-1):2*(N-1)+N-2]=ah*np.ones((N-2,1))
		bh[3*(N-1)-1:2*(N-1)+2*(N-2)]=-al*np.ones((N-2,1))

		# Real-time Initials

		[Xj,Vj] = self.getSurroundVehicleState(obs+1)
		[relateXj,relateVj]=self.getSurroundVehicleRelateState(obs+1)
		#velocity = self.getVelocity()
		velocityVec=self.getVelocityVector()
		#vehicleDirection = self.vehicle.sensor.getSelfAngle()
		pos=self.getPos()

		x0=pos[1]
		#x0=0	
		be[0]=x0
		v0=velocityVec[1]
		#v0=20	
		be[1]=v0
		xobs0=Xj[1]
		#xobs0=50	
		xobsv0=Vj[1]
		#xobsv0=22	
		margin=ACCHead*xobsv0

		xobs=matrix(np.arange(xobs0,xobs0+xobsv0*Ts*N,xobsv0*Ts))		
		ub=xobs-margin*matrix(np.ones((N,1)))
		#Ah[2*(N-1)+2*(N-2):2*(N-1)+2*(N-2)+N,:]=np.eye(N)
		#bh[2*(N-1)+2*(N-2):2*(N-1)+2*(N-2)+N]=ub
		Ah=matrix(Ah)
		bh=matrix(bh) 
		Ae=matrix(Ae)
		be=matrix(be)
		V=matrix(V)
		QV=matrix(QV)
		A=matrix(A)
		QA=matrix(QA)
		Qobs=matrix(Qobs)
	
		if vr>=xobsv0:
		    H=V.trans()*QV*V+A.trans()*QA*A+Qobs
		    f1=-V.trans()*QV*vr*matrix(np.ones((N-1,1)))
		    f2=margin*matrix(np.ones((N,1)))
		    f3=xobs-f2
		    f4=-a3*f3
		    f=matrix(f1+f4)
		else:
		    H=V.trans()*QV*V+A.trans()*QA*A;
		    f=-V.trans()*QV*vr*matrix(np.ones((N-1,1)))
	
		x=solvers.qp(H,f,Ah,bh,Ae,be)
		dv1=A*x['x']
		dv=dv1[0]
		return dv
	if obs==-1:
		refInput = self.previewController()
		return refInput[0]
    
    def laneSelection(self):
	vSurr=[0,0,0,0]
	vMax=0
	x0=self.getPos()[1]
	isempty=[1,1,1,1]
        for index in range(self.numSurrounding): 
	    [Xj,Vj] = self.getSurroundVehicleState(index+1)
	    Id=self.getSurrVehicleLaneId(index+1)
	    vSurr[Id]=Vj[1]
	    if Xj[1]>x0:
		isempty[Id]=0
            if Vj[1]>vMax:
		vMax=Vj[1]
		fastLane=Id
	selfLane=self.getCurrLaneId()
	if isempty[selfLane]==1:
		desiredLane=selfLane
	else:
		if selfLane!=fastLane:
			desiredLane=fastLane
		else:
			desiredLane=selfLane
			if selfLane>0 and selfLane<3:
				if isempty[selfLane-1]==1:
			    		desiredLane=selfLane-1
				elif isempty[selfLane+1]==1:
			    		desiredLane=selfLane+1
			if selfLane==0:
				if isempty[selfLane+1]==1:
			    		desiredLane=selfLane+1
			if selfLane==3:
				if isempty[selfLane-1]==1:
			    		desiredLane=selfLane-1
	return [desiredLane,vSurr]


    def getSurrVehicles(self,lane=1):
	surrV=[]
	for index in range(self.numSurrounding):
	    if self.getSurrVehicleLaneId(index+1)==lane:
		surrV.append(index)
	return surrV
    
    def laneChangeController(self):
	LCHead=0.3
	LCDis=100
	[desiredLane,vSurr]=self.laneSelection()
	selfLane=self.getCurrLaneId()
	if desiredLane>selfLane:
	    for index in self.getSurrVehicles(selfLane+1):
		[relateXj,relateVj]=self.getSurroundVehicleRelateState(index+1)
		if abs(relateXj[1])<LCDis:
		    LCDis=abs(relateXj[1])
	    if LCDis>10+vSurr[selfLane+1]*LCHead:
		self.targetLane=selfLane+1
	if desiredLane<selfLane:
	    for index in self.getSurrVehicles(selfLane-1):
		[relateXj,relateVj]=self.getSurroundVehicleRelateState(index+1)
		if abs(relateXj[1])<LCDis:
		    LCDis=abs(relateXj[1])
	    if LCDis>10+vSurr[selfLane-1]*LCHead:
		self.targetLane=selfLane-1   

    '''def iLQR(self):
	self.targetLane
	velocity = self.getVelocity()
	pos=self.getPos()
	path = []
	N=length(path)
	xdim=4
	udim=2
	Ts=0.1
	iterations=100

	# Initials
	P=zeros(xdim,xdim,N)
	b=zeros(N,xdim)
	c=zeros(N,1)
	d=zeros(N-1,2)
	e=zeros(N-1,1)
	A=zeros(4,4,N-1)
	B=zeros(4,2,N-1)
	f=1000
	Qf=[f 0 0 0;0 f 0 0;0 0 0 0;0 0 0 100*f]
	c1=1;a1=1;a2=1;a3=0;a4=0
	Q=c1*diag([a1 a2 a3 a4])
	c2=100
	R=c2*[1 0;0 1]
	J=zeros(iterations,1)
	x=zeros(xdim,N)
	u=zeros(udim,N-1)
	up=zeros(udim,N-1)
	du=zeros(udim,N-1)
	x0=[-5;0;10;0]
	x(:,1)=x0
	xgoal=zeros(xdim,N)
	xgoal(1:2,:)=path
	xgoal(4,end)=0
	P(:,:,N)=Qf
	dx=zeros(xdim,N)
	cost=10000000
	flag=0
	step=1
	# The Iterations
	for j=1:iterations:
	# Forward Simulation
	    for i=2:N:
		vk=x(3,i-1);thk=x(4,i-1);dvk=u(1,i-1);dthk=u(2,i-1);thm=thk+dthk*Ts
		if dthk!=0:
		    x(1,i)=x(1,i-1)+vk*((sin(thm)-sin(thk))/dthk)+dvk*(cos(thm)+dthk*Ts*sin(thm)-cos(thk))/dthk^2
		    x(2,i)=x(2,i-1)+(dvk*sin(thm)-dvk*sin(thk))/dthk^2+(vk*cos(thk)-(vk+dvk*Ts)*cos(thk+dthk*Ts))/dthk
		else:
		    x(1,i)=x(1,i-1)+cos(thk)*(vk*Ts+dvk/2*Ts^2)
		    x(2,i)=x(2,i-1)+sin(thk)*(vk*Ts+dvk/2*Ts^2)
		x(3,i)=vk+dvk*Ts
		x(4,i)=thk+dthk*Ts

	# LQR
	    xd=xgoal-x
	    b(N,:)=-xd(:,N)'*Qf
	    c(N)=1/2*xd(:,N)'*Qf*xd(:,N)
	    for k=N:-1:2:
		ud=-u
		vk=x(3,k-1)
		thk=x(4,k-1)
		dvk=u(1,k-1)
		dthk=u(2,k-1)
		thm=thk+dthk*Ts
		
		if dthk!=0:
		    dxdth=vk/dthk*(cos(thm)-cos(thk))+dvk/dthk^2*(-sin(thm)+sin(thk)+dthk*Ts*cos(thm))
		    dydth=dvk*Ts/dthk*sin(thm)+vk/dthk*(sin(thm)-sin(thk))+dvk/dthk^2*(cos(thm)-cos(thk))
		    dxddv=(cos(thm)+dthk*Ts*sin(thm)-cos(thk))/dthk^2
		    dxddth=vk*((sin(thk)-sin(thm))/dthk^2+Ts*cos(thm)/dthk)-2*dvk*(cos(thm)-cos(thk)+dthk*Ts*sin(thm))/dthk^3+dvk*Ts^2*cos(thm)/dthk
		    dyddv=(sin(thm)-sin(thk))/dthk^2-Ts*cos(thm)/dthk
		    dyddth=2*dvk*(sin(thk)-sin(thm))/dthk^3+(cos(thm)*(vk+Ts*dvk)-vk*cos(thk))/dthk^2+Ts*sin(thm)*(vk+Ts*dvk)/dthk+dvk*Ts*cos(thm)/dthk^2
		    Ak=[1 0 (sin(thk+dthk*Ts)-sin(thk))/dthk dxdth;
		                0 1 (cos(thk)-cos(thm))/dthk dydth;
		                0 0 1 0;0 0 0 1]
		    Bk=[dxddv dxddth;dyddv dyddth;Ts 0;0 Ts]
		else:
		    Ak=[1 0 Ts*cos(thk) -vk*Ts*sin(thk);
		                0 1 Ts*sin(thk) vk*Ts*cos(thk);
		                0 0 1 0;0 0 0 1]
		    Bk=[cos(thk)*Ts^2/2 0;sin(thk)*Ts^2/2 0;Ts 0;0 Ts]
		A(:,:,k-1)=Ak
                B(:,:,k-1)=Bk
	        Pk=P(:,:,k)
	        d(k-1,:)=-ud(:,k-1)'*R
	        e(k-1)=1/2*ud(:,k-1)'*R*ud(:,k-1)
	        invB=inv(Bk'*Pk*Bk+R)
	        P(:,:,k-1)=-Ak'*Pk*Bk*invB*Bk'*Pk*Ak+Ak'*Pk*Ak+Q
	        b(k-1,:)=b(k,:)*Ak-xd(:,k-1)'*Q-(b(k,:)*Bk+d(k-1,:))*invB*Bk'*Pk*Ak
	        c(k-1)=c(k)+e(k-1)+1/2*xd(:,k-1)'*Q*xd(:,k-1)-1/2*(b(k,:)*Bk+d(k-1,:))*invB*(b(k,:)*Bk+d(k-1,:))'
	    for k=1:N-1:
		du(:,k)=-inv(B(:,:,k)'*P(:,:,k+1)*B(:,:,k)+R)*(B(:,:,k)'*P(:,:,k+1)*A(:,:,k)*dx(:,k)+B(:,:,k)'*b(k,:)'+d(k,:)')
		dx(:,k+1)=A(:,:,k)*dx(:,k)+B(:,:,k)*du(:,k)
	    u=u+du'''                                 

    def doControl(self,lf=1):
        #print(self.previewController())
        #print(self.getVelocity())
	self.laneChangeController()
        refInput = self.previewController()
        dtheta=refInput[1]
	dThMargin=1
	if dtheta>dThMargin:
	    dtheta=dThMargin
        if dtheta<-dThMargin:
	    dtheta=-dThMargin
        return [self.ACC(),dtheta,0]

# the planningAgent use safety controller to achieve collision avoidance
class planningAgent(autoBrakeAgent):
    def __init__(self,vGain=50,thetaGain=20,desiredV=25,laneId=0,num=1,radiu=500,headway=20):
        self.vGain=vGain
        self.thetaGain=thetaGain
        self.desiredV=desiredV
        self.safeHeadway = headway
        self.traj = [[0,0]]
        self.numSurrounding = num # number of surrounding vehicles
        self.h=2
        self.dmin = 10
        self.ts = 1.00/15
        self.targetLane=laneId
        self.changeFlag=0
        self.phiFlag=0
        self.frontObs=0
        self.radiu=radiu
        self.Range=15
        self.alpha=50
        self.ita=2
        self.changeThres=1.3
        self.previousInput=[0,0]
        self.timeStep=0
        safetyCommand = file('safetyCommand.txt', 'w')
        vProf = file('velocity.txt', 'w')
        dProf = file('distance.txt', 'w')

    # The Efficiency controller
    def getTrajectory(self):
        #return self.getPreview(1,25)

        horizon = 20
        refTraj = []
        if len(self.traj)>1:
            refTraj = self.traj[:]

        preTraj = self.getPreview(self.getCurrLaneId(),horizon)
        #print(self.getCurrLaneId())
        if len(self.traj)>1:
            i = len(self.traj)
        else:
            i = len(self.traj)-1

        while i < horizon:
            i = i+1
            refTraj.append(preTraj[i])
        #print(len(refTraj))

        x0 = self.getPos()

        # get trajectory of surrounding vehicles
        obs = []
        for i in range(self.numSurrounding):
            obs.append(self.getPrediction(i,horizon))
        #print(obs)
        traj = opt.CFS_FirstOrder(x0,refTraj,obs,horizon,self.ts)
        return traj


    # Get lateral distance to a trajectory; return (dis, angle, index)
    def getDisAngle(self,x0,theta,traj):
        l1 = np.linalg.norm(utility.substruct(traj[0],x0))
        for index in range(len(traj)-1):
            l2 = np.linalg.norm(utility.substruct(traj[index+1],x0))
            direction = utility.substruct(traj[index+1],traj[index])
            angle = utility.vec2ang(direction)

            if l1**2+l2**2 < np.linalg.norm(direction)**2 or l1 < l2:
                return (utility.perpLength(utility.substruct(x0,traj[index]),direction),(theta-angle+math.pi) % (2*math.pi)-math.pi,index)


    # Overwrite the preview controller in Preview Agent
    def previewController(self):

        '''h = self.h

        if len(self.traj)<h:
            self.traj = list(self.getTrajectory())'''

        self.traj = self.getPreview(self.targetLane,25)
        futureTraj = self.traj
        #print(futureTraj)

        ff = [0,0]#self.getFeedforwardControl(futureTraj)
        
        #dis, angle, index = self.getDisAngle(self.getPos(),utility.vec2ang(self.getDirection()),futureTraj)
        diffPosV=self.vehicle.sensor.getCordVelocity(futureTraj)
        fb = self.getFeedbackControl(self.getAngle(),self.getDis(self.targetLane),diffPosV)
        #print(dis-self.getDis(),angle-self.getAngle())

        '''i = 0
        while i <= index:
            self.traj.pop(0)
            i += 1'''

        acceleration = ff[0]+fb[0]
        steerV = ff[1]+fb[1]

        steeringLimit=45

        if steerV>steeringLimit:
          steerV=steeringLimit
        if steerV<-steeringLimit:
          steerV=-steeringLimit

        return [acceleration,steerV,0]

    def getCurrLaneId(self):
        '''pos=self.getPos()
        if pos[0]<-3.6:
            return 0
        if pos[0]<0:
            return 1
        if pos[0]<3.6:
            return 2
        return 3'''

        dev=-self.vehicle.sensor.getCordPos(0)[0]-6
        if dev<-4:
            return 0
        if dev<0:
            return 1
        if dev<4:
            return 2
        return 3 

    def getSurrVehicleLaneId(self,num):
        '''x=self.getSurroundVehicleState(num)[0][0]
        if x<-3.6:
            return 0
        if x<0:
            return 1
        if x<3.6:
            return 2
        return 3'''
        dev=-self.vehicle.sensor.getSurroundVehicle(num).sensor.getCordPos(0)[0]-6
        if dev<-4:
            return 0
        if dev<0:
            return 1
        if dev<4:
            return 2
        return 3 


    def getLaneDirection(self):
        futureTraj = self.getPreview(1,2)
        laneDirection = np.array([futureTraj[1][0]-futureTraj[0][0],futureTraj[1][1]-futureTraj[0][1]])
        laneDirection = laneDirection/np.linalg.norm(laneDirection)
        return laneDirection

    # Prediction of the future trajectory of surrounding vehicles
    # Here we use constant speed model
    def getPrediction(self,id,horizon):
        [x0,v0] = self.getSurroundVehicleState(id)

        traj = np.zeros((horizon, 2))
        for i in range(horizon):
            traj[i] = x0+v0*self.ts*(i+1)*0.1
        #print(traj)
        return traj

    def changeTargetLane(self):
        self.targetLane=self.getCurrLaneId()

    def safetyController(self):
        self.frontObs=0
        Lstack, Sstack = [], []
        velocity = self.getVelocity()
        velocityVec=self.getVelocityVector()
        vehicleDirection = self.vehicle.sensor.getSelfAngle()
        pos=self.getPos()
        laneDirection =  self.getLaneDirection()
        currLaneObsId=[]
        for index in range(self.numSurrounding):
            if self.getSurrVehicleLaneId(index+1)==self.targetLane:
                currLaneObsId.append(index)
        d=1000
        for index in currLaneObsId:
            [relateX,relateV]=self.getSurroundVehicleRelateState(index+1)
            if np.linalg.norm(relateX)<d:
                d=np.linalg.norm(relateX)
                obs=index

        #the Time
        self.timeStep=self.timeStep+1
        time=self.timeStep/60
        
        #write dProf
        dProf = open('distance.txt', 'a')
        dProf.write(str(time)+'\t'+str(d)+'\n')
        
        currLaneObsId=[obs]
        #print currLaneObsId
        self.phiFlag=0
        for index in currLaneObsId:

            [Xj,Vj] = self.getSurroundVehicleState(index+1)
            [relateXj,relateVj]=self.getSurroundVehicleRelateState(index+1)
            x0=pos[0]
            y0=pos[1]
            v0=velocity
            x0V=velocityVec[0]
            y0V=velocityVec[1]
            xj=Xj[0]
            yj=Xj[1]
            xjV=Vj[0]
            yjV=Vj[1]
            vj=np.sqrt(xjV**2+yjV**2)
            dx=relateXj[0]
            dy=relateXj[1]
            dxV=relateVj[0]
            dyV=relateVj[1]
            d=np.sqrt(dx**2+dy**2)
            dV=(relateXj[0]*relateVj[0]+relateXj[1]*relateVj[1])/d
            phix0=2*dx-self.alpha*(dx*(dxV*dx+dyV*dy)/d**3-dxV/d)
            phixj=-2*dx+self.alpha*(dx*(dxV*dx+dyV*dy)/d**3-dxV/d)
            phiy0=2*dy-self.alpha*(dy*(dxV*dx+dyV*dy)/d**3-dyV/d)
            phiyj=-2*dy+self.alpha*(dy*(dxV*dx+dyV*dy)/d**3-dyV/d) 
            phiv0=self.alpha*(x0V*dx+y0V*dy)/(v0*d)
            phivj=-self.alpha*(xjV*dx+yjV*dy)/(vj*d)    
            phitheta0=-self.alpha*(y0V*dx-x0V*dy)/d
            phithetaj=self.alpha*(yjV*dx-xjV*dy)/d 
            L=[phiv0,phitheta0]  
            phi=self.dmin**2-d**2-self.alpha*dV
            S=-phi-(phixj*xjV+phiyj*yjV)-(phix0*x0V+phiy0*y0V) 

            
            if phi>0 and d<self.Range:
                Lstack.append(L)
                Sstack.append(S)
                self.phiFlag=1
                self.changeFlag=1
                self.frontObs=1

        if self.changeFlag==1:
            '''if self.getCurrLaneId()!=self.targetLane:
                self.changeTargetLane()
                self.changeFlag=0'''
            if -self.vehicle.sensor.getCordPos(self.getCurrLaneId())[0]>self.changeThres:
                self.targetLane=self.getCurrLaneId()+1
            if -self.vehicle.sensor.getCordPos(self.getCurrLaneId())[0]<-self.changeThres:
                self.targetLane=self.getCurrLaneId()-1
            self.changeFlag=0

        if self.getCurrLaneId()==0:
            if self.frontObs==1:
                D=10
            else:
                D=2
            alpha=2
            thetaRel=self.vehicle.sensor.getCordAngle()
            sinTheRel=math.sin(thetaRel)
            cosTheRel=math.cos(thetaRel)
            x=-self.vehicle.sensor.getCordPos(0)[0]
            v=velocity
            r=self.radiu
            L=[alpha*sinTheRel,alpha*v*cosTheRel]
            w=4
            phi=D-w**2/4-x**2-w*x+alpha*v*sinTheRel
            #S=-5-(2*x+w)*v*sinTheRel+alpha*v**2/r*(1-cosTheRel)   # for curve lane
            S=-5-(2*x+w)*v*sinTheRel  # for straight lane

            if phi>0:
                self.phiFlag=1
                Lstack.append(L)
                Sstack.append(S)

        if self.getCurrLaneId()==3:
            if self.frontObs==1:
                D=10
            else:
                D=2
            alpha=2
            thetaRel=self.vehicle.sensor.getCordAngle()
            sinTheRel=math.sin(thetaRel)
            cosTheRel=math.cos(thetaRel)
            x=-self.vehicle.sensor.getCordPos(3)[0]
            v=velocity
            r=self.radiu
            L=[-alpha*sinTheRel,-alpha*v*cosTheRel]
            w=4
            phi=D-w**2/4-x**2+w*x-alpha*v*sinTheRel
            #S=-5-(2*x+w)*v*sinTheRel+alpha*v**2/r*(sinTheRel+1)   # for curve lane
            S=-5-(2*x-w)*v*sinTheRel  # for straight lane
            #print phi
            
            if phi>0:
                self.phiFlag=1
                Lstack.append(L)
                Sstack.append(S)

        refInput = self.previewController()
        ang=refInput[1]*np.pi/180 #transfer to rad
        refU=[refInput[0],np.tan(ang)*velocity/2.1] #transfer from steer to thetaV
        if refU[1]>0.5:
            refU[1]=0.5
        if refU[1]<-0.5:
            refU[1]=-0.5
        
        # The saturation for stability
        Lstack.append([0, 1])
        Sstack.append(0.5)
        Lstack.append([0, -1])
        Sstack.append(0.5)

        vdot0=self.previousInput[0]
        thetadot0=self.previousInput[1]
        dvMax=5
        dthetaMax=0.3
        dthetaMax1=0.3
        a=10
        b=10
        c=0.1
        d=0.1
        
        if self.phiFlag==1:
            Lcheck=copy.copy(Lstack)
            Lcheck.append([1,0])
            Lcheck.append([-1,0])
            Lcheck.append([0,1])
            Lcheck.append([0,-1])
            Scheck=copy.copy(Sstack)
            Scheck.append(vdot0+dvMax)
            Scheck.append(-vdot0+dvMax)
            Scheck.append(thetadot0+dthetaMax)
            Scheck.append(-thetadot0+dthetaMax)
            Lcheck=matrix(Lcheck,(2,len(Lcheck)),'d')
            Lcheck=Lcheck.trans()
            Scheck=matrix(Scheck,(len(Scheck),1),'d')
            Q=matrix([5, 0, 0, 5],(2,2),'d')
            p = matrix(refU[0:2],(2,1),'d')
            p=Q*p
            # Check if Us and Uf intersects
            '''c=matrix([1,1],(2,1),'d')
            sol = solvers.conelp(c, Lcheck, Scheck)
            if sol['status']=='optimal':
                sol = solvers.qp(Q,-p, Lcheck,Scheck)
                newU=sol['x']
            elif sol['status']=='primal infeasible' or sol['status']=='unknown':
                z=[]
                for j in range(len(Lstack)):
                    z.append(0)
                    z.append(0)
                z=matrix(z,(len(Lstack),2),'d')
                Lstack.append([0,0])
                Lstack.append([0,0])
                Lstack.append([0,0])
                Lstack.append([0,0])
                Lstack=matrix(Lstack,(2,len(Lstack)),'d')
                Lstack = Lstack.trans()
                Lplus=matrix([1,-1,0,0,0,0,1,-1],(4,2),'d')
                Lfull=matrix([[Lstack],[matrix([z,Lplus])]])
                Q=matrix([a, 0, -a, 0,0,b,0,-b,-a,0,a,0,0,-b,0,b],(4,4),'d')
                q=matrix([0,0,0,0],(4,1),'d')
                sol = solvers.qp(Q,q, Lfull,Scheck)
                newU=sol['x'][0:2]'''

            z=[]
            for j in range(len(Lstack)):
                z.append(0)
                z.append(0)
            z=matrix(z,(len(Lstack),2),'d')
            Lstack.append([0,0])
            Lstack.append([0,0])
            Lstack.append([0,0])
            Lstack.append([0,0])
            Lstack=matrix(Lstack,(2,len(Lstack)),'d')
            Lstack = Lstack.trans()
            Lplus=matrix([1,-1,0,0,0,0,1,-1],(4,2),'d')
            Lfull=matrix([[Lstack],[matrix([z,Lplus])]])
            #Q=matrix([a, 0, -a, 0,0,b,0,-b,-a,0,a,0,0,-b,0,b],(4,4),'d')
            Q=matrix([a, 0, -a, 0,0,b,0,-b,-a,0,a+c,0,0,-b,0,b+d],(4,4),'d')
            #q=matrix([0,0,0,0],(4,1),'d')
            q=matrix([0,0,-2*c*refU[0],-2*d*refU[1]],(4,1),'d')
            sol = solvers.qp(Q,q, Lfull,Scheck)
            newU=sol['x'][2:4]
            dis=matrix(sol['x'],(4,1),'d').trans()*Q*matrix(sol['x'],(4,1),'d')
                
        else:
            Lsafe=matrix([1,-1,0,0,0,0,1,-1],(4,2),'d')
            Ssafe=matrix([vdot0+dvMax,-vdot0+dvMax,thetadot0+dthetaMax1,-thetadot0+dthetaMax1],(4,1),'d')
            #Ssafe=matrix([vdot0+dvMax,-vdot0+dvMax,0.00002,0.00002],(4,1),'d')
            Q=matrix([5, 0, 0, 5],(2,2),'d')
            p = matrix(refU[0:2],(2,1),'d')
            p=Q*p
            sol = solvers.qp(Q,-p, Lsafe,Ssafe)
            newU=sol['x']
            #newU=refU[0:2]
            #print refU[0:2]

        '''Lstack = matrix(Lstack,(2,len(Lstack)),'d')
        Lstack = Lstack.trans()
        Sstack = matrix(Sstack,(len(Sstack),1),'d')

        #Q = matrix([2, 1, 1, 1000],(2,2),'d')
        Q = matrix([5, 0, 0, 1000],(2,2),'d')
        p = matrix(refU[0:2],(2,1),'d')
        p = Q*p
        #sol = solvers.qp(Q,-p)#, Lstack,Sstack)
        sol = solvers.qp(Q,-p, Lstack,Sstack)
        newU = sol['x']'''
        #print refU
        newInput=copy.copy(newU)

        #check safety controller enabling
        contrEnab=0
        if abs(newInput[0]-refU[0])>0.01 and abs(newInput[1]-refU[1])>0.01:
            contrEnab=1
        #get velocity vector
        latV=self.vehicle.sensor.getCordVelocity(self.getPreview(self.targetLane,1))
        longiV=self.vehicle.sensor.getCordLongiVelocity(self.getPreview(self.targetLane,1))
        
        safetyCommand = open('safetyCommand.txt', 'a')
        safetyCommand.write(str(time)+'\t'+str(contrEnab)+'\n')
        vProf = open('velocity.txt', 'a')
        vProf.write(str(time)+'\t'+str(latV)+'\t'+str(longiV)+'\n')
        
        newInput[1]=np.arctan(newInput[1]/velocity*2.1) #transfer from thetaV to steer
        newInput[1]=newInput[1]*180/np.pi #transfer to degree
        self.previousInput=copy.copy(newU)
        
        
        return [newInput[0],newInput[1],0]

    def doControl(self,lf=1):
        #print(self.previewController())
        #print(self.getVelocity())
        return self.safetyController()

# the learningAgent learns the movement of surrounding vehicles
class learningAgent(planningAgent):
    def __init__(self,vGain=50,thetaGain=20,desiredV=15,headway=20):
        self.vGain=vGain
        self.thetaGain=thetaGain
        self.desiredV=desiredV
        self.safeHeadway = headway
        self.traj = [0,0]
        self.numSurrounding = 1 # number of surrounding vehicles
        self.h=15
        self.dmin = 10

    # data should include all the information of surrounding vehicles
    def perception(data):
        # processs data to get information at current time step
        self.obs.append(data)
        for vehicle in data:
            if vehicle not in self.surrounding.keys():
                # Initialization of the prediction
                self.surrounding[vehicle]=[{'intention':[1,0,0,0,0],
                                           'par':[[1,1,1],
                                                  [1,1,1,1,1],
                                                  [1,1,1,1,1]],
                                           'action':0}]
            self.surrounding[vehicle].append(self.getFeature(data[vehicle]))

    def getAllSurroundingVehicleState(self):
        for i in range(self.numSurrounding):
            self.getSurroundVehicleState()

    def getTrafficFlowSpeed(self):
        return 15

    def getFeature(self,data):
        F = {}

        # f1: Longitudinal accelaration


        # f2: Deccelaration light


        # f3: Turn signal


        # f4: Speed relative to the traffic flow
        F['f4']=-self.getTrafficFlowSpeed()

        # f5: Speed relative to the front vehicle


        # f6: current lane id


        # f7: current lane clearance


        # f8: lateral velocity


        # f9: lateral devation from the center of its current lane


        # f10: lateral deviation from the center of its target lane

        return F

    def infer(self,data):
        # to be completed
        return [1,0,0,0,0]

    def learning(self):
        # Behavior Classification
        for vehicle in self.surrounding.keys():
            if self.surrounding[vehicle][-1]['f6'] == -1: # If executing a maneuver
                self.surrounding[vehicle][-1]['intention'] = [0,0,0,0,0]
                max_index = self.surrounding[vehicle][-2]['action']
                self.surrounding[vehicle][-1]['intention'][max_index] = 1
                self.surrounding[vehicle][-1]['action'] = max_index
            else:
                # If the maneuver is completed
                if self.surrounding[vehicle][-2]['intention'][1] == 0:
                    self.surrounding[vehicle][-1]['intention'] = [1,0,0,0,0]
                    self.surrounding[vehicle][-1]['action'] = 0
                else:
                    # Inference based on probability model
                    self.surrounding[vehicle][-1]['intention'] = self.infer(self.surrounding[vehicle][-2:0])
                    max_value = max(self.surrounding[vehicle][-1]['intention'])
                    max_index = self.surrounding[vehicle][-1]['intention'].index(max_value)
                    self.surrounding[vehicle][-1]['action'] = max_index
            if  self.surrounding[vehicle][-1]['action'] ==  self.surrounding[vehicle][-2]['action']:
                self.surrounding[vehicle][-1]['par'] = PAA( self.surrounding[vehicle][-2], self.surrounding[vehicle][-1], self.surrounding[vehicle][-1]['action'])

    def getPrediction():

        return 0

    def doControl(self):
        return self.safetyController()



class densoAgent(learningAgent):

    def doControl(self):
        return [0,0]
