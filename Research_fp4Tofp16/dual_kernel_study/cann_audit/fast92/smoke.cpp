#include <acl/acl.h>
#include <cstdio>
int main() {
  unsigned a=aclInit(nullptr); printf("aclInit=%u\n",a); if(a) return 2;
  unsigned b=aclrtSetDevice(0); printf("aclrtSetDevice=%u\n",b); if(b) return 3;
  printf("aclrtGetSocName=%s\n",aclrtGetSocName());
  aclrtResetDevice(0);aclFinalize();return 0;
}
