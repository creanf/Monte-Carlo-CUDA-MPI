#ifndef TIMING_H
#define TIMING_H

typedef unsigned long long ticks;


//-------------------------------------------------COMMENT OUT IF RUNNING ON SUPER COMPUTER and UNCOMMENT OTHER
/*static inline ticks getticks(void)
{
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}*/

// IBM POWER9 System clock with 512MHZ resolution.
static __inline__ ticks getticks(void)
 {
   unsigned int tbl, tbu0, tbu1;

   do {
     __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
     __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
     __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
   } while (tbu0 != tbu1);

   return (((unsigned long long)tbu0) << 32) | tbl;
 }


#endif
