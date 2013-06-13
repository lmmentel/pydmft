!===============================================================================
! Test Program for module reading the one and two -electron integrals from 
! gamess-us files.
!===============================================================================
!
program test 
 use dictionary
  implicit none
  real(8), allocatable :: occ(:)
  character(len=100)   :: filename
  integer(8)           :: i, nrec, length
!
!! specify the dictionary filename
    filename = "LiH_cc-pvtz_LiH_cc-pvtz_3.0_NO.F10"
!
    nrec = 21
    length = 50

    allocate(occ(length))

    call ReadReals(filename, occ, nrec)
!
    do i = 1, length
        write(*,*) i ,occ(i)
    enddo

    deallocate(occ)
end program
