module dictionary

implicit none

contains

 subroutine ReadReals(filename, vector, record)
  implicit none
  integer,     parameter  :: ik = 8

  character(len=*),      intent(in)    :: filename
  real(8), dimension(:), intent(inout) :: vector
  integer(ik),           intent(in)    :: record 

  integer(ik), parameter :: idaf   = 10
  integer(ik), parameter :: irecln = 4090

  integer(ik)                 :: irecst, is, ipk
  integer(ik), dimension(950) :: ioda, ifilen


    open(unit=idaf,file=trim(filename),access='direct',recl=8*irecln,form='unformatted')
    read(idaf,rec=1) irecst, ioda, ifilen, is, ipk

    call daread(idaf, record, ioda, ifilen, irecln, vector, int(size(vector), kind=ik))

    close(unit=idaf)
 end subroutine ReadReals

 subroutine daread(idaf, nrec, ioda, ifilen, irecln, vector, length)
  implicit none

  integer,     parameter  :: ik = 8
  integer(ik), intent(in) :: idaf, length, nrec, irecln

  real(8),     dimension(length), intent(inout) :: vector
  integer(ik), dimension(950),    intent(in)    :: ioda, ifilen

! local variables

  integer(ik) :: n, ns, is, nsp, fi, lent, lenw   

    n = ioda(nrec)

    if (n == -1) then
        write(*,'(/1X,"ERROR *** ATTEMPTING A BOGUS READ OF A DAF RECORD."/ &
   &              1X,"RECORD NUMBER",I5," OF LENGTH",I10, " WAS NEVER PREVIOUSLY WRITTEN.")') nrec, length
        stop "exiting"
    endif

    if (length <= 0 ) then 
        write(*,'(/1X,"ERROR *** ATTEMPTING A BOGUS READ OF A DAF RECORD."/ &
   &              1X,"RECORD NUMBER",I5," OF LENGTH",I10," HAS NO LENGTH.")') nrec, length
        stop "exiting"
    endif

    if (length > ifilen(nrec)) then
        write(*,'(/1X,"ERROR *** ATTEMPTING A BOGUS READ OF A DAF RECORD."/      &
   &              1X,"ATTEMPTING TO READ",I10," WORDS FROM RECORD NUMBER",I5/    &
   &              1X,"BUT THIS RECORD WAS PREVIOUSLY WRITTEN WITH ONLY",         &
   &              I10," WORDS.")') length, nrec, ifilen(nrec)
        stop "exiting"
    endif

    IS = -IRECLN + 1
    NS = N
    LENT = length
    do while (lent >= 1) 
        IS = IS + IRECLN
        fi = IS + LENT - 1
        if ((fi-IS+1) .GT. IRECLN) fi = IS + IRECLN - 1
        NSP = NS
        LENW = fi - IS + 1
        read(unit=idaf, rec=nsp) vector
        LENT = LENT - IRECLN
        NS = NS + 1
        N = NS
    enddo
 end subroutine daread 

end module dictionary

!===============================================================================
! Test Program
!===============================================================================
!
!program test 
! use dictionary
!  implicit none
!  real(8), dimension(4) :: occ
!  character(len=100)   :: filename, recname
!  integer(8)           :: i, nrec, length
!
!! specify the dictionary filename
!    filename = "h2_midi_0.7.F10"
!
!    nrec = 21
!    recname = 'occupations'
!    length = 4
!
!    call ReadReals(filename, occ, recname)
!
!    do i = 1, length
!        write(*,*) i ,occ(i)
!    enddo
!end program
