#ifndef OUTFLOW_PRESSURE_H_
#define OUTFLOW_PRESSURE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>

class OutFlowPressure
{
public:
    // grav is supposed to be positive
    OutFlowPressure (amrex::Geometry const& geom, amrex::Real grav,
                     amrex::Array<bool,AMREX_SPACEDIM*2> const& is_outflow);
    ~OutFlowPressure ();

    // rho must have at least one ghost cell
    void computePressure (amrex::MultiFab& pres, amrex::MultiFab const& rho);

private:
    amrex::Real m_deltaz;
    amrex::Real m_fac;
    amrex::MultiFab m_p;
    amrex::MultiFab m_rho;
};

#endif
