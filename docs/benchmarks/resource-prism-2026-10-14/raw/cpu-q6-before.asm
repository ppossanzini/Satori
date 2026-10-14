; Assembly listing for method Satori.Core.QuantizedKAvx2Kernel:DotProductQ6K(ptr,ptr):int (FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; FullOpts code
; optimized code
; rbp based frame
; partially interruptible
; No PGO data
; 0 inlinees with PGO data; 2 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 264
       lea      rbp, [rsp+0x130]
 
G_M000_IG02:                ;; offset=0x0019
       mov      rbx, rdi
       lea      r15, [rbx+0x80]
       lea      r14, [rbx+0xC0]
       lea      r13, [rsi+0x04]
       lea      r12, [rsi+0x104]
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       vmovups  ymmword ptr [rbp-0x50], ymm0
       vbroadcastss ymm1, dword ptr [reloc @RWD04]
       vmovups  ymmword ptr [rbp-0x70], ymm1
       vbroadcastss ymm2, dword ptr [reloc @RWD08]
       vmovups  ymmword ptr [rbp-0x90], ymm2
       vxorps   ymm3, ymm3, ymm3
       vmovups  ymmword ptr [rbp-0xB0], ymm3
       xor      eax, eax
       mov      dword ptr [rbp-0xB4], eax
       xor      ecx, ecx
       jmp      SHORT G_M000_IG04
 
G_M000_IG03:                ;; offset=0x007A
       mov      ecx, eax
 
G_M000_IG04:                ;; offset=0x007C
       mov      edi, ecx
       sar      edi, 31
       and      edi, 3
       add      edi, ecx
       sar      edi, 2
       mov      esi, ecx
       sar      esi, 31
       and      esi, 3
       add      esi, ecx
       and      esi, -4
       mov      dword ptr [rbp-0xB8], ecx
       mov      edx, ecx
       sub      edx, esi
       mov      esi, edi
       shl      esi, 5
       movsxd   rsi, esi
       add      rsi, r15
       shl      edi, 6
       movsxd   rdi, edi
       add      rdi, rbx
       mov      r8d, edx
       shr      r8d, 31
       add      r8d, edx
       and      r8d, -2
       mov      r9d, edx
       sub      r9d, r8d
       shl      r9d, 5
       movsxd   r8, r9d
       vmovups  ymm4, ymmword ptr [rdi+r8]
       cmp      edx, 2
       jl       SHORT G_M000_IG06
 
G_M000_IG05:                ;; offset=0x00DA
       vpsrlw   ymm4, ymm4, 4
       vpand    ymm4, ymm4, ymm0
       vmovups  ymmword ptr [rbp-0xF0], ymm4
       jmp      SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x00ED
       vpand    ymm4, ymm4, ymm0
       vmovups  ymmword ptr [rbp-0xF0], ymm4
 
G_M000_IG07:                ;; offset=0x00F9
       vmovups  ymm5, ymmword ptr [rsi]
       vmovups  ymmword ptr [rsp], ymm5
       lea      rdi, [rbp-0x110]
       mov      esi, edx
       call     [Satori.Core.QuantizedKAvx2Kernel:Q6KHighField(System.Runtime.Intrinsics.Vector256`1[byte],int):System.Runtime.Intrinsics.Vector256`1[byte]]
       vmovups  ymm1, ymmword ptr [rbp-0x70]
       vpand    ymm0, ymm1, ymmword ptr [rbp-0x110]
       vpsllw   ymm0, ymm0, 4
       vmovups  ymm2, ymmword ptr [rbp-0x90]
       vpand    ymm0, ymm0, ymm2
       vpor     ymm0, ymm0, ymmword ptr [rbp-0xF0]
       mov      eax, dword ptr [rbp-0xB8]
       mov      ecx, eax
       shl      ecx, 5
       movsxd   rcx, ecx
       vpmaddubsw ymm0, ymm0, ymmword ptr [rcx+r13]
       lea      ecx, [rax+rax]
       movsxd   rdx, ecx
       movsx    rdi, byte  ptr [r14+rdx]
       inc      ecx
       movsxd   rcx, ecx
       movsx    rsi, byte  ptr [r14+rcx]
       vmovd    xmm3, edi
       vpinsrw  xmm3, xmm3, edi, 1
       vpinsrw  xmm3, xmm3, edi, 2
       vpinsrw  xmm3, xmm3, edi, 3
       vpinsrw  xmm3, xmm3, edi, 4
       vpinsrw  xmm3, xmm3, edi, 5
       vpinsrw  xmm3, xmm3, edi, 6
       vpinsrw  xmm3, xmm3, edi, 7
       vmovd    xmm4, esi
       vpinsrw  xmm4, xmm4, esi, 1
       vpinsrw  xmm4, xmm4, esi, 2
       vpinsrw  xmm4, xmm4, esi, 3
       vpinsrw  xmm4, xmm4, esi, 4
       vpinsrw  xmm4, xmm4, esi, 5
       vpinsrw  xmm4, xmm4, esi, 6
       vpinsrw  xmm4, xmm4, esi, 7
       vinserti128 ymm3, ymm3, xmm4
       vpmaddwd ymm0, ymm0, ymm3
       vpaddd   ymm3, ymm0, ymmword ptr [rbp-0xB0]
       movsx    rdx, word  ptr [r12+2*rdx]
       imul     edx, edi
       movsx    rcx, word  ptr [r12+2*rcx]
       imul     ecx, esi
       add      ecx, edx
       shl      ecx, 5
       add      ecx, dword ptr [rbp-0xB4]
       inc      eax
       cmp      eax, 8
       mov      dword ptr [rbp-0xB4], ecx
       vmovups  ymmword ptr [rbp-0xB0], ymm3
       vmovups  ymm0, ymmword ptr [rbp-0x50]
       jl       G_M000_IG03
 
G_M000_IG08:                ;; offset=0x01F9
       vmovups  ymm3, ymmword ptr [rbp-0xB0]
       vmovaps  ymm0, ymm3
       vextracti128 xmm1, ymm3
       vpaddd   xmm0, xmm1, xmm0
       vpshufd  xmm1, xmm0, 78
       vpaddd   xmm0, xmm1, xmm0
       vpshufd  xmm1, xmm0, -79
       vpaddd   xmm0, xmm1, xmm0
       vmovd    ecx, xmm0
       sub      ecx, dword ptr [rbp-0xB4]
       mov      eax, ecx
 
G_M000_IG09:                ;; offset=0x022D
       vzeroupper 
       add      rsp, 264
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
RWD00  	dd	0F0F0F0Fh		; 7.05334e-30
RWD04  	dd	03030303h		; 3.85009e-37
RWD08  	dd	30303030h		; 6.40969e-10

; Total bytes of code 578

