module integrals 
 
implicit none

contains


   integer(8) function address(i,j,k,l)                                       
   integer(8) :: i,j,k,l,ij,kl                                                
                                                                        
    ij = max(i,j)*(max(i,j)-1)/2 + min(i,j)                                
    kl = max(k,l)*(max(k,l)-1)/2 + min(k,l)                                
                                                                        
    address = max(ij,kl)*(max(ij,kl)-1)/2 + min(ij,kl)                     
                                                                        
  end function

  subroutine ReadInAO(TwoIntAO, filename)
! read two-electron integrals over AO's
  integer,     parameter :: IK = kind(1)
  integer(IK), parameter :: LK2 = kind(2)
  integer(IK), parameter :: LK8 = kind(8)
  integer(IK), parameter :: SP = kind(1.0)
  integer(IK), parameter :: DP = kind(1.0D0)
  integer(IK), parameter :: IPgamess = 8  !4 = 32 bit integer, 8 = 64 bit integer
  integer(IK), parameter :: DPgamess = DP
  integer(IK), parameter :: LKgamess = 8
  integer(IK), parameter :: twoein = 9

    real(DP),         intent(inout)   :: TwoIntAO(:)
    character(len=*), intent(in)      :: filename

    real(DP),          allocatable  :: buffer(:)
    integer(IPgamess), allocatable  :: indexBuffer(:)

    real(DP)          :: temp

    integer(IK)       :: readStatus
    integer(IK)       :: m
    integer(IPgamess)       :: i,j,k,l
    integer(IK)       :: bufLength, twoIntIndexBufSize, twoIntBufferSize
    integer(IPgamess) :: label, label1, label2
    integer(IPgamess) :: length, nintmx, labsiz, twoeao
    logical           :: largeLabels    

    nintmx = 15000                               ! gamess parameter controling read buffer
    labsiz = 1                                   ! gamess parameter controling the index range, for maxao > 256 set to 2  
    twoeao = twoein

    if (labsiz /= 1_IPgamess .and. labsiz /= 2_IPgamess) then
      write(*,*) 'RdTwoIntAO:  CONFUSION IN LABSIZ! '
      stop
    endif

    largeLabels = (labsiz == 2_IPgamess)

    twoIntBufferSize = int(nintmx, kind=IK)

    twoIntIndexBufSize = twoIntBufferSize
    if (largeLabels) then
      if (IPgamess == 4) twoIntIndexBufSize = 2*twoIntIndexBufSize
    else
      if (IPgamess == 8) twoIntIndexBufSize = (twoIntIndexBufSize + 1) / 2
    endif

    open(unit=twoeao, file=trim(filename), status='old', form='unformatted')
    rewind(twoeao)

    allocate(buffer(twoIntBufferSize))
    allocate(indexBuffer(twoIntIndexBufSize))

    length = 1_IPgamess
    do while (length > 0_IPgamess)
      Read(twoeao,iostat=readStatus) length,indexBuffer,buffer 
      if (readStatus /= 0) then
        if (readStatus == 2) then
          write(*,*) 'RdTwoIntAO: ENCOUNTERED UNEXPECTED END WHILE READING TWO-ELECTRON FILE'
        else
          write(*,*) 'RdTwoIntAO: ENCOUNTERED I/O ERROR WHILE READING TWO-ELECTRON FILE'
        endif
        stop
      endif

      bufLength = abs(int(length, kind=IK))
      if (bufLength > twoIntBufferSize) stop

      do m=1, bufLength
        if (IPgamess == 4) then                  ! 32-bit Gamess integers
          if (largeLabels) then       
            label1 = indexbuffer(2*m-1) 
            label2 = indexBuffer(2*m)
            i = ishft(label1, -16_IPgamess)
            j = iand( label1,  65535_IPgamess)
            k = ishft(label2, -16_IPgamess)
            l = iand( label2,  65535_IPgamess)
          else
            label = indexBuffer(m)
            i =      ishft(label, -24_IPgamess)
            j = iand(ishft(label, -16_IPgamess), 255_IPgamess)
            k = iand(ishft(label,  -8_IPgamess), 255_IPgamess)
            l = iand(      label,                255_IPgamess)
          endif  
        else                                     ! 64-bit Gamess integers
          if (largeLabels) then
            label = indexBuffer(m)
            i = int(     ishft(label,   -48_IPgamess),                  kind=IK)
            j = int(iand(ishft(label,   -32_IPgamess), 65535_IPgamess), kind=IK)
            k = int(iand(ishft(label,   -16_IPgamess), 65535_IPgamess), kind=IK)
            l = int(iand(      label,                  65535_IPgamess), kind=IK)  
          else
            if (mod(m,2) == 0) then
              label = indexBuffer(m/2)
              i = int(iand(ishft(label, -24_IPgamess ), 255_IPgamess), kind=IK)
              j = int(iand(ishft(label, -16_IPgamess ), 255_IPgamess), kind=IK)
              k = int(iand(ishft(label,  -8_IPgamess ), 255_IPgamess), kind=IK)
              l = int(iand(      label,                 255_IPgamess), kind=IK)
            else
              label = indexBuffer(m/2+1)
              i = int(     ishft(label, -56_IPgamess),                kind=IK)
              j = int(iand(ishft(label, -48_IPgamess), 255_IPgamess), kind=IK)
              k = int(iand(ishft(label, -40_IPgamess), 255_IPgamess), kind=IK)
              l = int(iand(ishft(label, -32_IPgamess), 255_IPgamess), kind=IK)
            endif      
          endif
        endif

        temp = buffer(m)
        TwoIntAO(address(i,j,k,l)) = temp
 
      enddo
    enddo
   
    deallocate(buffer)
    deallocate(indexBuffer)
    close(twoeao)
  end subroutine

  subroutine ReadInMO(TwoIntMO, filename)
!   use VarModule
  integer,     parameter :: IK = kind(1)
  integer(IK), parameter :: LK2 = kind(2)
  integer(IK), parameter :: LK8 = kind(8)
  integer(IK), parameter :: SP = kind(1.0)
  integer(IK), parameter :: DP = kind(1.0D0)
  integer(IK), parameter :: IPgamess = 8  !4 = 32 bit integer, 8 = 64 bit integer
  integer(IK), parameter :: DPgamess = DP
  integer(IK), parameter :: LKgamess = 8
  integer(IK), parameter :: twoein = 9

    real(DP),         intent(inout)   :: TwoIntMO(:)
    character(len=*), intent(in)      :: filename

    real(DP),          allocatable  :: buffer(:)
    integer(IPgamess), allocatable  :: indexBuffer(:)

    real(DP)          :: temp

    integer(IK)       :: readStatus
    integer(IK)       :: m
    integer(IPgamess)       :: i,j,k,l
    integer(IK)       :: bufLength, twoIntIndexBufSize, twoIntBufferSize
    integer(IPgamess) :: label, label1, label2
    integer(IPgamess) :: length, nintmx, labsiz, twoemo
    logical           :: largeLabels    

    nintmx = 15000
    labsiz = 1 
    twoemo = twoein

    if (labsiz /= 1_IPgamess .and. labsiz /= 2_IPgamess) then
      write(*,*) 'RdTwoIntAO:  CONFUSION IN LABSIZ! '
      stop
    endif

    largeLabels = (labsiz == 2_IPgamess)

    twoIntBufferSize = int(nintmx, kind=IK)

    twoIntIndexBufSize = twoIntBufferSize
    if (largeLabels) then
      if (IPgamess == 4) twoIntIndexBufSize = 2*twoIntIndexBufSize
    else
      if (IPgamess == 8) twoIntIndexBufSize = (twoIntIndexBufSize + 1) / 2
    endif

    open(unit=twoemo, file=trim(filename), status='old', form='unformatted')
    rewind(twoemo)
    read(twoemo)

    allocate(buffer(twoIntBufferSize))
    allocate(indexBuffer(twoIntIndexBufSize))

    length = 1_IPgamess
    do while (length > 0_IPgamess)
      Read(twoemo,iostat=readStatus) length,indexBuffer,buffer 
      if (readStatus /= 0) then
        if (readStatus == 2) then
          write(*,*) 'RdTwoIntMO: ENCOUNTERED UNEXPECTED END WHILE READING TWO-ELECTRON FILE'
        else
          write(*,*) 'RdTwoIntMO: ENCOUNTERED I/O ERROR WHILE READING TWO-ELECTRON FILE'
        endif
        stop
      endif

      bufLength = abs(int(length, kind=IK))
      if (bufLength > twoIntBufferSize) stop

      do m=1, bufLength
        if (IPgamess == 4) then                  ! 32-bit Gamess integers
          if (largeLabels) then       
            label1 = indexbuffer(2*m-1) 
            label2 = indexBuffer(2*m)
            i = ishft(label1, -16_IPgamess)
            j = iand( label1,  65535_IPgamess)
            k = ishft(label2, -16_IPgamess)
            l = iand( label2,  65535_IPgamess)
          else
            label = indexBuffer(m)
            i =      ishft(label, -24_IPgamess)
            j = iand(ishft(label, -16_IPgamess), 255_IPgamess)
            k = iand(ishft(label,  -8_IPgamess), 255_IPgamess)
            l = iand(      label,                255_IPgamess)
          endif  
        else                                     ! 64-bit Gamess integers
          if (largeLabels) then
            label = indexBuffer(m)
            i = int(     ishft(label,   -48_IPgamess),                  kind=IK)
            j = int(iand(ishft(label,   -32_IPgamess), 65535_IPgamess), kind=IK)
            k = int(iand(ishft(label,   -16_IPgamess), 65535_IPgamess), kind=IK)
            l = int(iand(      label,                  65535_IPgamess), kind=IK)  
          else
            if (mod(m,2) == 0) then
              label = indexBuffer(m/2)
              i = int(iand(ishft(label, -24_IPgamess ), 255_IPgamess), kind=IK)
              j = int(iand(ishft(label, -16_IPgamess ), 255_IPgamess), kind=IK)
              k = int(iand(ishft(label,  -8_IPgamess ), 255_IPgamess), kind=IK)
              l = int(iand(      label,                 255_IPgamess), kind=IK)
            else
              label = indexBuffer(m/2+1)
              i = int(     ishft(label, -56_IPgamess),                kind=IK)
              j = int(iand(ishft(label, -48_IPgamess), 255_IPgamess), kind=IK)
              k = int(iand(ishft(label, -40_IPgamess), 255_IPgamess), kind=IK)
              l = int(iand(ishft(label, -32_IPgamess), 255_IPgamess), kind=IK)
            endif      
          endif
        endif

        temp = buffer(m)
        TwoIntMO(address(i,j,k,l)) = temp
 
      enddo
    enddo
   
    deallocate(buffer)
    deallocate(indexBuffer)
    close(twoemo)
  end subroutine

!  subroutine print4indexVec(fi, nb)
 !  use dmftHelpModule
!  integer(IPgamess) :: i,j,k,l,ij,kl, nb
!  real(DP), intent(in)  :: fi(:)

!  write(*,*)
!   ij = 0
!   do i = 1,nb
!     do j = 1,i
!       ij = ij+1
!       kl = 0
!       do k = 1,nb
!         do l = 1,k
!           kl = kl + 1
!             if(ij >= kl .and. abs(fi(address(i,j,k,l))) > 1.0e-10) write(*,'(2x,4i5,2x,f20.14)') i,j,k,l,fi(address(i,j,k,l))
!         enddo
!       enddo 
!     enddo 
!   enddo
!
!  end subroutine 

end module integrals
