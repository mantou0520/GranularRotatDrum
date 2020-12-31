/************************************************************************
 * MechSys - Open Library for Mechanical Systems                        *
 *                                                                      *
 * This program is free software: you can redistribute it and/or modify *
 * it under the terms of the GNU General Public License as published by *
 * the Free Software Foundation, either version 3 of the License, or    *
 * any later version.                                                   *
 *                                                                      *
 * This program is distributed in the hope that it will be useful,      *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the         *
 * GNU General Public License for more details.                         *
 *                                                                      *
 * You should have received a copy of the GNU General Public License    *
 * along with this program. If not, see <http://www.gnu.org/licenses/>  *
 ************************************************************************/

#include <math.h>
#include <random>

#include <gsl/gsl_linalg.h>
// MechSys
#include <mechsys/dem/domain.h>
#include <mechsys/util/fatal.h>
#include <mechsys/util/util.h>
#include <mechsys/mesh/unstructured.h>
#include <mechsys/mesh/structured.h>
#include <mechsys/linalg/matvec.h>

using std::cout;
using std::endl;

double DrumR = 15.0;         // the radius of the drum, [cm]
double DrumW = 2.0;          // the width of the drum, [cm]
double Cx = 0.0;             // x coordinate of the center of the rotating drum
double Cy = 0.0;             // y coordinate of the center of the rotating drum
double Cz = 0.0;             // z coordinate of the center of the rotating drum
// center of the rotating drum
Vec3_t DrumCenter(Cx,Cy,Cz);
int Nf = 50;                 // Number of faces to build the drum
int StepPerOut;              // Number of steps per output of gravitational acceleration
double halfp = 0.5;          // position on the half of the segment, don't change this.
double Rcorr = DrumR*cos(2*M_PI*(halfp)/Nf); // corrected radius of the circle due to the 
                                        // number of segments employed to make the drum
double clockw,AngV,AngAc,StopAc;

void Report(DEM::Domain & dom, void * UD)  //function that is going to be called every time step
{
	// UserData & dat = (*static_cast<UserData *>(UD));
 
	String ff;
	ff.Printf    ("%s_bf_%04d",dom.FileKey.CStr(), dom.idx_out);
	dom.WriteBF(ff.CStr());
}

void Setup (DEM::Domain & domi, void * UD) //function that is going to be called every time step
{
	// TIME SET THE CONSTANT ANGULAR VELOCITY OF THE ROTATING DRUM
    // double pos2 = 0.5*AngAc*pow(StopAc-Tswitch,2) + AngV*(domi.Time-StopAc);
    double rotatAngle = AngV*(domi.Time);
	Vec3_t FCentri;
	Vec3_t FCorioli;
	Vec3_t FGrav;
	// Vec3_t VelRel; // relative velocity
	// Vec3_t GravAcc = (0.0, -981.00*sin(rotatAngle), -981.00*cos(rotatAngle));
	int StepNum = domi.Time/domi.Dt;
	if (StepNum%StepPerOut==0.0)
	{
		std::cout << "Time = " << domi.Time << "; Gx = " << 0.0 << "; Gy=" << -981.00*sin(rotatAngle) << "; Gz=" << -981.00*cos(rotatAngle) << std::endl; 
	}
	
    #pragma omp parallel for schedule (static) num_threads(domi.Nproc)
	for (size_t i = 0; i<domi.Particles.Size(); i++)
	{
		if (domi.Particles[i]->Tag < 0)
		{
			FCorioli[0] = 0.0;
			FCorioli[1] = -AngV*(domi.Particles[i]->v[2])*domi.Particles[i]->Props.m*2.0;
			FCorioli[2] = AngV*(domi.Particles[i]->v[1])*domi.Particles[i]->Props.m*2.0;
			FCentri[0] = 0.0;
			FCentri[1] = domi.Particles[i]->Props.m*AngV*AngV*domi.Particles[i]->x[1];
			FCentri[2] = domi.Particles[i]->Props.m*AngV*AngV*domi.Particles[i]->x[2];
			FGrav[0] = 0.0;
			FGrav[1] = domi.Particles[i]->Props.m*(-981.00*sin(rotatAngle));
			FGrav[2] = domi.Particles[i]->Props.m*(-981.00*cos(rotatAngle));
			domi.Particles[i]->Ff = FGrav + FCentri - FCorioli;
		}
	}
}

void SetupFluid (DEM::Domain & domi, void * UD) //function that is going to be called every time step
{
	
}


int main(int argc, char **argv) try
{   
	if (argc<2) throw new Fatal("This program must be called with one argument: the name of the data input file without the '.inp' suffix.\nExample:\t %s filekey\n",argv[0]);
	
	// Setting number of CPUs
	size_t Nproc = 1;
	if (argc>=3) Nproc = atoi(argv[2]);
	
	String filekey  (argv[1]);
	String filename (filekey+".inp");
	if (!Util::FileExists(filename)) throw new Fatal("File <%s> not found",filename.CStr());
	ifstream infile(filename.CStr());
    
	String CrossSection;  // Shape of the cross-section of the rotating drum
	String ptype;         // Particle type, sphere or voronoi
	String test;          // Test type, it is rotating drum test now
	bool   Cohesion;      // Decide if coheison is going to be simulated
	double fraction;      // Fraction of particles to be generated
	double Kn;            // Normal stiffness
	double Kt;            // Tangential stiffness
	double Gn;            // Normal dissipative coefficient
	double Gt;            // Tangential dissipative coefficient
	double Mu;            // Microscopic friction coefficient
	double Muw;           // Frictional coefficient of the bottom wall
	double Bn;            // Cohesion normal stiffness
	double Bt;            // Cohesion tangential stiffness
	double Bm;            // Cohesion torque stiffness
	double Beta;          // Rolling stiffness coefficient (only for spheres)
	double Eps;           // Threshold for breking bonds
    double SphR;          // average radius of a sphere
	double R;             // Spheroradius
	size_t seed;          // Seed of the ramdon generator
	double dt;            // Time step
	double dtOut1;        // Time step for output for the dropping stage
	double dtOut;         // Time step for output for the collapsing stage
	size_t scalingx;      // scalingx
	size_t scalingy;      // scalingy
	size_t scalingz;      // scalingz
	double rho;           // rho, particle density
	double AngW;          // Angular velocity of the rotating drum[rad/s]
	double Tf1;           // Final time for the dropping stage test
	double Tf;            // Final time for the collapsing test
	double RevTotal;      // total number of revolutions of the rotating drum
	double SavePerRev;    // number of data saving per revolution
	double Filling;       // filling ratio of particles in the drum
	double Large_v_Total; // ratio between large particles and small particles
	double SizeRatio;     // size ratio between large and small particles;
	double DensRatio;     // density ratio between large and small particles;
	{
		infile >> CrossSection;    infile.ignore(200,'\n');
		infile >> ptype;           infile.ignore(200,'\n');
		infile >> test;            infile.ignore(200,'\n');
		infile >> Cohesion;        infile.ignore(200,'\n');
		infile >> fraction;        infile.ignore(200,'\n');
		infile >> Kn;              infile.ignore(200,'\n');
		infile >> Kt;              infile.ignore(200,'\n');
		infile >> Gn;              infile.ignore(200,'\n');
		infile >> Gt;              infile.ignore(200,'\n');
		infile >> Mu;              infile.ignore(200,'\n');
		infile >> Muw;             infile.ignore(200,'\n');
		infile >> Bn;              infile.ignore(200,'\n');
		infile >> Bt;              infile.ignore(200,'\n');
		infile >> Bm;              infile.ignore(200,'\n');
		infile >> Beta;            infile.ignore(200,'\n');
		infile >> Eps;             infile.ignore(200,'\n');
        infile >> SphR;            infile.ignore(200,'\n');
		infile >> R;               infile.ignore(200,'\n');
		infile >> seed;            infile.ignore(200,'\n');
		infile >> dt;              infile.ignore(200,'\n');
		infile >> dtOut1;          infile.ignore(200,'\n');
		infile >> dtOut;           infile.ignore(200,'\n');
		infile >> scalingx;        infile.ignore(200,'\n');
		infile >> scalingy;        infile.ignore(200,'\n');
		infile >> scalingz;        infile.ignore(200,'\n');
		infile >> rho;             infile.ignore(200,'\n');
		infile >> AngW;            infile.ignore(200,'\n');
		infile >> Tf1;             infile.ignore(200,'\n');
		infile >> Tf;              infile.ignore(200,'\n');
		infile >> RevTotal;        infile.ignore(200,'\n');
		infile >> SavePerRev;      infile.ignore(200,'\n');
		infile >> Filling;         infile.ignore(200,'\n');
		infile >> Large_v_Total;   infile.ignore(200,'\n');
		infile >> SizeRatio;       infile.ignore(200,'\n');
		infile >> DensRatio;       infile.ignore(200,'\n');
	}

	clockw  = 1.0; // clockwise direction (clockw=-1) and counter clockwise direction (clockw=1)
    AngV    = clockw*(AngW);
    AngAc   = clockw*(M_PI*(1./10.));
	// Stage 1: dropping particles
	// domain
	DEM::Domain d;

	//Some key parameters
	
	double KnL = Kn/(scalingx*scalingy); //Stiffness constant for large particles
	double KtL = Kt/(scalingx*scalingy);
	double KnS = Kn/(4.0*scalingx*scalingy); //Stiffness constant for large particles
	double KtS = Kt/(4.0*scalingx*scalingy);
    //Generate particles
	// we have to decide which part of the volume is for generating particles
	double AreaParti = 1.6*Filling*M_PI*DrumR*DrumR;
	double LenParti = sqrt(AreaParti);
	double VolLarge = 0.0;
	double VolSmall = 0.0;
	// domain of small particles
    Vec3_t Xmin_1(-DrumW/2.0,-LenParti/2.0,-LenParti/2.0);
    Vec3_t Xmax_1(DrumW/2.0, -LenParti/2.0 + (1.0-1.2*Large_v_Total)*LenParti, LenParti/2.0);
	// domain of large particles
    Vec3_t Xmin_2(-DrumW/2.0, -LenParti/2.0 + (1.0-1.2*Large_v_Total)*LenParti,-LenParti/2.0);
    Vec3_t Xmax_2(DrumW/2.0, LenParti/2.0, LenParti/2.0);

    d.GenSpheresBox (-1, Xmin_1, Xmax_1, SphR, rho, "HCP", seed, fraction, 0.9);
    d.GenSpheresBox (-2, Xmin_2, Xmax_2, SizeRatio*SphR, DensRatio*rho, "HCP", seed, fraction, 0.9);

    // d.WriteXDMF("test");

	// Add all the DEM segments to make the outer case of the rotating drum
    double angle1,angle2,seglen,rcoo;
	double rhoWall = 1.2; // the density of the drum
    Vec3_t apoin,bpoin;
    Quaternion_t q;
    for(int i=0; i < Nf;++i)
	{
		angle1 = 2*M_PI*i/Nf;
		angle2 = 2*M_PI*(i+1)/Nf;
		rcoo = DrumR*cos((2*M_PI*0.5)/Nf);
		apoin = 0.0, DrumCenter(1)+rcoo*cos(angle1), DrumCenter(2)+rcoo*sin(angle1);
		bpoin = 0.0, DrumCenter(1)+rcoo*cos(angle2), DrumCenter(2)+rcoo*sin(angle2);
		// apoin=DrumCenter(0)+rcoo*cos(angle1), DrumCenter(1)+rcoo*sin(angle1), 0.0;
		// bpoin=DrumCenter(0)+rcoo*cos(angle2), DrumCenter(1)+rcoo*sin(angle2), 0.0;
		// seglen=norm(Vec3_t(DrumCenter(0)+DrumR*cos(angle1), DrumCenter(1)+DrumR*sin(angle1), 0.0)-Vec3_t(DrumCenter(0)+DrumR*cos(angle2), DrumCenter(1)+DrumR*sin(angle2), 0.0)); // get the length with no correction
		seglen = norm(Vec3_t(0.0, DrumCenter(1)+DrumR*cos(angle1), DrumCenter(2) + DrumR*sin(angle1)) - Vec3_t(0.0, DrumCenter(1)+DrumR*cos(angle2), DrumCenter(2) + DrumR*sin(angle2)));
    	if ( angle2>(M_PI/2.) && angle2<=((3./2.)*M_PI) && angle1>=(M_PI/2.) && angle1<((3./2.)*M_PI) )
		{
    		d.AddPlane(101+i, apoin,SphR, DrumW, seglen,rhoWall,0.0,&OrthoSys::e0); //left side of the circle
		}
    	else
		{
        	d.AddPlane(101+i , apoin,SphR, DrumW, seglen,rhoWall,0.0,&OrthoSys::e0); //right side of the circle
		}
		// q = (1.0, 0.0, 0.0, cos(0.5*(angle1+angle2)));
		std::cout << "seg number = " << i+1 << std::endl; 
		std::cout << "seg length = " << seglen << std::endl; 
		std::cout << "seg vert = " << apoin << std::endl; 
       	NormalizeRotation((angle1 + M_PI/2), OrthoSys::e0, q);
       	Vec3_t locrot = apoin;
       	d.GetParticle(101+i)->Rotate(q,locrot);
		d.GetParticle(101+i)->FixVeloc();
    }
	
	size_t countdel = 0;
    // set the particle properties
	for (size_t np=0;np<d.Particles.Size();np++)
    {
		d.Particles[np]->Ff = d.Particles[np]->Props.m*Vec3_t(0.0,0.0, -981.0);
		d.Particles[np]->Props.Gn = Gn; // restitution coefficient
		d.Particles[np]->Props.Mu = Mu; // frictional coefficient
		d.Particles[np]->Props.Kn = KnL;
		d.Particles[np]->Props.Kt = KtL;
		if (d.Particles[np]->Tag > 100)
		{
			d.Particles[np]->Props.Mu = Muw;  // set the friction of the drum
		}
		else if (d.Particles[np]->Tag == -1)
		{
			VolSmall = VolSmall + d.Particles[np]->Props.m/rho;
		}
		else if (d.Particles[np]->Tag == -2)
		{
			VolLarge = VolLarge + d.Particles[np]->Props.m/rho;
		}
	}

	for (size_t np=0; np<d.Particles.Size(); np++)
	{
		if (d.Particles[np]->Tag < 0 && d.Particles[np]->x[1]*d.Particles[np]->x[1] + d.Particles[np]->x[2]*d.Particles[np]->x[2] > (DrumR-3.0*SphR)*(DrumR-3.0*SphR))
		{
			countdel = countdel+1;
			d.Particles[np]->Tag = -3;
		}
	}

	Array<int> delpar0;

	if (countdel > 0)
	{
		delpar0.Push(-3);
		d.DelParticles(delpar0);
	}

	for (size_t np=0; np<d.Particles.Size(); np++)
	{
		if (d.Particles[np]->Tag == -1)
		{
			VolSmall = VolSmall + d.Particles[np]->Props.m/rho;
		}
		else if (d.Particles[np]->Tag == -2)
		{
			VolLarge = VolLarge + d.Particles[np]->Props.m/rho;
		}
	}

	double VolRatioCal = VolLarge/VolSmall;
	std::cout << "The volume ratio between large and small particles is: " << VolRatioCal << std::endl;

	Dict B1;
	B1.Set(-1,"Kn Kt Beta",KnS, KtS, Beta);
	B1.Set(-2,"Kn Kt Beta",KnL, KtL, Beta);
	d.SetProps(B1);

    // define the periodic boundary condition
    d.Xmax = 1.0;
	d.Xmin = -1.0;

	// d.WriteXDMF("test1");

	dt = 0.5*d.CriticalDt(); //Calculating time step
	d.Alpha = R; //Verlet distance
	d.Solve(/*tf*/Tf1, dt, /*dtOut*/dtOut1, NULL, &Report, "drop_spheres", 2, Nproc);

	d.Save("Stage_1");

	// Start the stage of rotating drum

	DEM::Domain dom;
	dom.Load("Stage_1");
    // reset the particle properties
	for (size_t np=0;np<d.Particles.Size();np++)
    {
		dom.Particles[np]->Ff = dom.Particles[np]->Props.m*Vec3_t(0.0,0.0, -981.0);
		dom.Particles[np]->Props.Gn = Gn; // restitution coefficient
		dom.Particles[np]->Props.Mu = Mu; // frictional coefficient
		dom.Particles[np]->Props.Kn = KnL;
		dom.Particles[np]->Props.Kt = KtL;
		if (dom.Particles[np]->Tag > 100)
		{
			dom.Particles[np]->Props.Mu = Muw;  // set the friction of the drum
		}
	}

	

	Dict B2;
	B2.Set(-1,"Kn Kt Beta",KnS, KtS, Beta);
	B2.Set(-2,"Kn Kt Beta",KnL, KtL, Beta);
	dom.SetProps(B2);

	for(int i=0; i < Nf;++i)
	{
		dom.GetParticle(101+i)->FixVeloc();
    }

	// define the periodic boundary condition
    dom.Xmax = 1.0;
	dom.Xmin = -1.0;

	// dom.WriteXDMF("test2");
	// setting up the output rate and the total running time
	dt = 0.5*dom.CriticalDt(); //Calculating time step
	dom.Alpha = R; //Verlet distance
	Tf = RevTotal*2.0*M_PI/AngW;
	dtOut = 1.0*2.0*M_PI/AngW/SavePerRev;
	StepPerOut = dtOut/5.0/dt;
	std::cout << "Total running time = " << Tf << std::endl; 
	std::cout << "Angular velocity = " << AngW << std::endl;
	std::cout << "Total number of revolutions = " << RevTotal << std::endl;
	std::cout << "Savings per revolution = " << SavePerRev << std::endl;
	std::cout << "dtOut of saving = " << dtOut << " second per saving" << std::endl;
	
	dom.Solve(/*tf*/Tf, dt, /*dtOut*/dtOut, &Setup, &Report, "rotat", 2, Nproc);

}
MECHSYS_CATCH
