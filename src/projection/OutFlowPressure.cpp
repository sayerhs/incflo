#include "OutFlowPressure.H"
#include <AMReX_Scan.H>
#include <algorithm>

using namespace amrex;

OutFlowPressure::OutFlowPressure (Geometry const& geom, Real grav,
                                  Array<bool,AMREX_SPACEDIM*2> const& is_outflow)
    : m_deltaz(geom.CellSize(2)), m_fac(geom.CellSize(2)*grav / 300.0)
{
    BoxList bl_p(IndexType::TheNodeType());
    BoxList bl_rho(IndexType::TheCellType());
    Box nd_domain = amrex::convert(geom.Domain(),IntVect::TheNodeVector());
    constexpr int nslices = 4;

    if (is_outflow[Orientation(Direction::x,Orientation::low)]) {
        Box const box = amrex::bdryLo(nd_domain, 0);
        int jmin = box.smallEnd(1);
        int jmax = box.bigEnd(1);
        for (int j = jmin; j <= jmax; j += nslices) {
            Box b = box;
            b.setSmall(1,j);
            b.setBig(1,std::min(j+nslices-1,jmax));
            bl_p.push_back(b);
            bl_rho.push_back(b.grow(IntVect(1,1,0)).growLo(2,1).enclosedCells());
        }
    }

    if (is_outflow[Orientation(Direction::x,Orientation::high)]) {
        Box const box = amrex::bdryHi(nd_domain, 0);
        int jmin = box.smallEnd(1);
        int jmax = box.bigEnd(1);
        for (int j = jmin; j <= jmax; j += nslices) {
            Box b = box;
            b.setSmall(1,j);
            b.setBig(1,std::min(j+nslices-1,jmax));
            bl_p.push_back(b);
            bl_rho.push_back(b.grow(IntVect(1,1,0)).growLo(2,1).enclosedCells());
        }
    }

    if (is_outflow[Orientation(Direction::y,Orientation::low)]) {
        Box const box = amrex::bdryLo(nd_domain, 1);
        int imin = box.smallEnd(0);
        int imax = box.bigEnd(0);
        for (int i = imin; i <= imax; i += nslices) {
            Box b = box;
            b.setSmall(0,i);
            b.setBig(0,std::min(i+nslices-1,imax));
            bl_p.push_back(b);
            bl_rho.push_back(b.grow(IntVect(1,1,0)).growLo(2,1).enclosedCells());
        }
    }

    if (is_outflow[Orientation(Direction::y,Orientation::high)]) {
        Box const box = amrex::bdryHi(nd_domain, 1);
        int imin = box.smallEnd(0);
        int imax = box.bigEnd(0);
        for (int i = imin; i <= imax; i += nslices) {
            Box b = box;
            b.setSmall(0,i);
            b.setBig(0,std::min(i+nslices-1,imax));
            bl_p.push_back(b);
            bl_rho.push_back(b.grow(IntVect(1,1,0)).growLo(2,1).enclosedCells());
        }
    }

    BoxArray ba_p(std::move(bl_p));
    BoxArray ba_rho(std::move(bl_rho));
    DistributionMapping dm{ba_p};
    m_p.define(ba_p, dm, 1, 0);
    m_rho.define(ba_rho, dm, 1, 0);
    m_p.setVal(0.0);
}

OutFlowPressure::~OutFlowPressure () {}

void OutFlowPressure::computePressure (MultiFab& pmf, MultiFab const& rhomf)
{
    m_rho.ParallelCopy(rhomf,0,0,1,1,0);
    amrex::Print() << "OutFlowPressure TEMPERATURE RANGE "
                   << m_rho.min(0) << " " << m_rho.max(0) << std::endl;
    amrex::Print()
        << "OutFlowPressure before calcs "
        << pmf.min(0) << " " << pmf.max(0)  << " "
        << m_p.min(0) << " " << m_p.max(0) << std::endl;

    Real fac = m_fac;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(m_p); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.validbox();
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        Array4<Real> const& p = m_p.array(mfi);
        Array4<Real const> const& rho = m_rho.const_array(mfi);
        for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
            int kmin = lo.z;
            int kmax = hi.z;
#ifdef AMREX_USE_GPU
            if (Gpu::inLaunchRegion()) {
                amrex::Scan::PrefixSum<Real>(kmax-kmin+1,
                    [=] AMREX_GPU_DEVICE (int kr) -> Real
                    {
                        int k = kmax-kr-1;
                        return fac*0.25*(rho(i-1,j-1,k)+rho(i,j-1,k)+rho(i-1,j,k)+rho(i,j,k));
                    },
                    [=] AMREX_GPU_DEVICE (int kr, Real const& x)
                    {
                        p(i,j,kmax-kr) = x;
                    },
                    Scan::Type::exclusive);
            } else
#endif
            {
                p(i,j,kmax) = 0.0;
                for (int k = kmax-1; k >= kmin; --k) {
#if 0
                    // Using actual temperature field
                    Real rhoavg = 0.25*(rho(i-1,j-1,k)+rho(i,j-1,k)+rho(i-1,j,k)+rho(i,j,k));

#else
                    // Hard-coding the exact temperature profile for testing
                    // m_probtype is 118
                    amrex::Real zco = k * m_deltaz;
                    amrex::Real rhoavg = 300.0 + zco * 0.01;
#endif
                    p(i,j,k) = p(i,j,k+1) + fac * (rhoavg - 300.0);
                } 
            }
        }}
    }

    pmf.ParallelCopy(m_p);
    amrex::Print()
        << "OutFlowPressure after integration "
        << pmf.min(0) << " " << pmf.max(0) << " "
        << m_p.min(0) << " " << m_p.max(0) << std::endl;
}
