C This program evaluates radially charge density on a unit circular disk (a=1.D0).
      program LP_disk

         implicit double precision (a-h,o-z)

         integer hit, infty
         integer nhit, i, ntrj, ninfty, nstep, nsteptotal

         double precision nstepavg
         double precision a
         double precision pi, pi2, twopi

         double precision start, stop
         integer iit(2), ftime

         common /flag/ hit, infty
         common /stat/ nhit, ninfty, nstep
         common /stattotal/ nsteptotal
         common /geometry/ a, d, theta
         common /intersect/ r_new
         common /angle/ pi, pi2, twopi
         common /charge_x/ i_x, i_x_max

         integer rndnum1, rndnum2
         external rndnum1, rndnum2
         external rnd1

C Initialization
         pi = dacos(-1.D0)
         pi2 = pi / 2.D0
         twopi = 2.D0 * pi
         ntrj = 100000000
C read *, ntrj
C a is the radius of a circular disk
         a = 1.0D0
         write(6, *) 'ntrj =', ntrj
C write(6, *) 'a =', a

C *****************************************************************
         ijsd1 = 51
         klsd1 = 211
         call rndini(ijsd1, klsd1)
         write(6, *) 'random number seed is ', ijsd1, klsd1

C *****************************************************************
         itt = ftime(iit)
         i_x_max = 100
         do 100 i = 100, 100
            i_x = i - 1
            call stattotalinitialization
            do 200 j = 1, ntrj
               call statinitialization
               call initialsitting(x, y, z)
               call lp_location(x, y, z)
               do while (hit .eq. 0 .and. infty .eq. 0)
                  call newstep(x, y, z)
                  call hit_or_not(x, y, z)
               end do
               call totalstat
  200       continue
C ***************statistics******************
            nstepavg = dble(nsteptotal) / dble(ntrj)
C write(6, *) 'nsteptotal =', nsteptotal, ' nstepavg =', nstepavg
            write(6, *) 'nhit =', nhit, ' ninfty =', ninfty
C d is the point where we calculate charge density.
C d is the distance from the center of the circular disk.
C a-d is the radius of the last passage sphere
            p = dble(ninfty) / dble(ntrj)
            charge_den = p * 3.D0 / (16.D0 * (a - d))

            r = a * dble(i_x) / dble(i_x_max)
            acharge = 1.D0 / (4.D0 * pi * dsqrt(1.D0 - r**2))
            write(6, *) r, charge_den, acharge
  100    continue

         itt = ftime(iit)
         stop = float(iit(1) + iit(2)) / 100.0
C write(*, *) 'running time =', stop - start
         stop
      end

C **********************************************
      subroutine stattotalinitialization

         implicit double precision (a-h,o-z)

         common /stattotal/ nsteptotal
         common /stat/ nhit, ninfty, nstep

         nhit = 0
         ninfty = 0
         nsteptotal = 0

         return
      end

C ***********************************************
      subroutine statinitialization

         implicit double precision (a-h,o-z)

         integer hit

         common /flag/ hit, infty
         common /stat/ nhit, ninfty, nstep

         hit = 0
         infty = 0
         nstep = 0

         return
      end

C ************************************************
      subroutine totalstat

         implicit double precision (a-h,o-z)

         common /stattotal/ nsteptotal
         common /stat/ nhit, ninfty, nstep

         nsteptotal = nsteptotal + nstep

         return
      end

C *************************************************
      subroutine initialsitting(x, y, z)

         implicit double precision (a-h,o-z)
         common /geometry/ a, d, theta
         common /angle/ pi, pi2, twopi
         common /charge_x/i_x, i_x_max

         d = a * dble(i_x) / dble(i_x_max)
         yyy = rnd1()

         phi0 = 2.D0 * pi * yyy
         x = d * dcos(phi0)
         y = d * dsin(phi0)
         z = 0.D0

         return
      end

C ***************************************************
      subroutine lp_location(x, y, z)

         implicit double precision (a-h,o-z)

         common /geometry/ a, d, theta
         common /angle/ pi, pi2, twopi
         external rnd1

         xxx = rnd1()
C If I choose uniform points on the hemisphere
C   theta = dacos(xxx)
C If I use usual double integration using Monte Carlo
C   theta is chosen uniformly between 0 and pi/2
C   theta = pi2 * xxx
C If I use dacos(dsqrt(xxx))
         theta = dacos(dsqrt(xxx))

         yyy = rnd1()
         phi = 2.D0 * pi * yyy

C a_d is the radius of last-passage sphere.
C Usually, a_d is the distance between the edge of the circular disk and
C the position where we want charge density.
         a_d = a - d
         x = x + a_d * dsin(theta) * dcos(phi)
         y = y + a_d * dsin(theta) * dsin(phi)
         z = z + a_d * dcos(theta)
         return
      end

C *****************************************************
      subroutine hit_or_not(x, y, z)

         implicit double precision (a-h,o-z)

         integer hit

         common /flag/ hit, infty
         common /stat/ nhit, ninfty, nstep
         common /geometry/ a, d, theta
         common /intersect/ r_new

         r_xy = dsqrt(x**2 + y**2)
         r = dsqrt(x**2 + y**2 + z**2)
         if (r .gt. a) then
            call infinity(r)
            if (infty .eq. 0) call bsurf(x, y, z, r)
         else
            hit = 1
            nhit = nhit + 1
         endif

         return
      end

C ******************** New Step ************************
      subroutine newstep(x, y, z)

         implicit double precision(a-h,o-z)

         common /angle/ pi, pi2, twopi

         xxx = rnd1()
         rs = sample(xxx)
         yyy = rnd1()
         phi1 = twopi * yyy

         d_r = dabs(z)
         x = x + rs * d_r * dcos(phi1)
         y = y + rs * d_r * dsin(phi1)
         z = 0.D0

         return
      end

C ***** Put the particle back to the b surface. ***********
      subroutine bsurf(x, y, z, r)

         implicit double precision(a-h,o-z)

         common /stat/ nhit, ninfty, nstep
         common /geometry/ a, d, theta
         common /angle/ pi, pi2, twopi
         external rnd1

         tmp = a / r
         p2 = rnd1()
         costh = - (1.D0 - tmp)**2 + 2.D0 * (1.D0 - tmp) * (1.D0 +
     *      tmp**2) * p2 + 2.D0 * tmp * (1.D0 + tmp**2) * p2**2
         costh = costh / (1.D0 - tmp + 2.D0 * tmp * p2) * 2
         sinth = dsqrt(dabs(1.D0 - costh**2))

         yyg = rnd1()
         phi = 2.D0 * pi * yyg
         cosph = dcos(phi)
         sinph = dsin(phi)

         xold = x
         yold = y
         zold = z
         p = dsqrt(xold**2 + yold**2)
         x = a * (sinth * cosph * xold * zold / (p * r)
     *      - sinth * sinph * yold / p + costh * xold / r)
         y = a * (sinth * cosph * yold * zold / (p * r)
     *      + sinth * sinph * xold / p + costh * yold / r)
         z = a * (-sinth * cosph * p / r + costh * zold / r)

         nstep = nstep + 1

         return
      end

C ***** Goes to infinity? *****************************
      subroutine infinity(r)

         implicit double precision(a-h,o-z)

         integer hit

         common /flag/ hit, infty
         common /stat/ nhit, ninfty, nstep
         common /geometry/ a, d, theta
         external rnd1

         ff = rnd1()
         ffr = ff * r
         if (ffr .ge. a) then
            infty = 1
            ninfty = ninfty + 1
         endif

         return
      end

C *****************************************************
      double precision function sample(xxx)
         implicit double precision(a-h,o-z)
         sample = dsqrt((1.D0 - xxx**2) / xxx**2)
         return
      end
C *****************************************************
