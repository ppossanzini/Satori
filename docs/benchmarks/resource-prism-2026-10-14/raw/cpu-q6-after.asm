; Assembly listing for method Satori.Core.QuantizedKAvx2Kernel:DotProductQ6K(ptr,ptr):int (FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; FullOpts code
; optimized code
; rbp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 3 single block inlinees; 2 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     rbx
       lea      rbp, [rsp+0x18]
 
G_M000_IG02:                ;; offset=0x000B
       lea      rax, [rdi+0x80]
       lea      rcx, [rdi+0xC0]
       lea      rdx, [rsi+0x04]
       add      rsi, 260
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       vbroadcastss ymm1, dword ptr [reloc @RWD04]
       vbroadcastss ymm2, dword ptr [reloc @RWD08]
       vxorps   ymm3, ymm3, ymm3
       xor      r8d, r8d
       xor      r9d, r9d
       align    [0 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0049
       mov      r10d, r9d
       sar      r10d, 31
       and      r10d, 3
       add      r10d, r9d
       sar      r10d, 2
       mov      r11d, r9d
       sar      r11d, 31
       and      r11d, 3
       add      r11d, r9d
       and      r11d, -4
       mov      ebx, r9d
       sub      ebx, r11d
       mov      r11d, r10d
       shl      r11d, 5
       movsxd   r11, r11d
       add      r11, rax
       shl      r10d, 6
       movsxd   r10, r10d
       add      r10, rdi
       mov      r15d, ebx
       shr      r15d, 31
       add      r15d, ebx
       and      r15d, -2
       mov      r14d, ebx
       sub      r14d, r15d
       shl      r14d, 5
       movsxd   r15, r14d
       vmovups  ymm4, ymmword ptr [r10+r15]
       cmp      ebx, 2
       jl       SHORT G_M000_IG05
 
G_M000_IG04:                ;; offset=0x00B0
       vpsrlw   ymm4, ymm4, 4
       vpand    ymm4, ymm4, ymm0
       jmp      SHORT G_M000_IG06
 
G_M000_IG05:                ;; offset=0x00BB
       vpand    ymm4, ymm4, ymm0
 
G_M000_IG06:                ;; offset=0x00BF
       vmovups  ymm5, ymmword ptr [r11]
       cmp      ebx, 2
       ja       SHORT G_M000_IG11
 
G_M000_IG07:                ;; offset=0x00C9
       mov      r10d, ebx
       lea      r11, [reloc @RWD12]
       mov      r11d, dword ptr [r11+4*r10]
       lea      rbx, G_M000_IG02
       add      r11, rbx
       jmp      r11
 
G_M000_IG08:                ;; offset=0x00E4
       vpsrlw   ymm5, ymm5, 4
       jmp      SHORT G_M000_IG12
 
G_M000_IG09:                ;; offset=0x00EB
       vpsrlw   ymm5, ymm5, 2
       jmp      SHORT G_M000_IG12
 
G_M000_IG10:                ;; offset=0x00F2
       jmp      SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x00F4
       vpsrlw   ymm5, ymm5, 6
 
G_M000_IG12:                ;; offset=0x00F9
       vpand    ymm5, ymm5, ymm1
       vpsllw   ymm5, ymm5, 4
       vpand    ymm5, ymm5, ymm2
       vpor     ymm4, ymm5, ymm4
       mov      r10d, r9d
       shl      r10d, 5
       movsxd   r10, r10d
       vpmaddubsw ymm4, ymm4, ymmword ptr [rdx+r10]
       lea      r10d, [r9+r9]
       movsxd   r11, r10d
       movsx    rbx, byte  ptr [rcx+r11]
       inc      r10d
       movsxd   r10, r10d
       movsx    r15, byte  ptr [rcx+r10]
       vmovd    xmm5, ebx
       vpbroadcastw xmm5, xmm5
       vmovd    xmm6, r15d
       vpbroadcastw xmm6, xmm6
       vinserti128 ymm5, ymm5, xmm6
       vpmaddwd ymm4, ymm4, ymm5
       vpaddd   ymm3, ymm4, ymm3
       movsx    r11, word  ptr [rsi+2*r11]
       imul     r11d, ebx
       movsx    r10, word  ptr [rsi+2*r10]
       imul     r10d, r15d
       add      r10d, r11d
       shl      r10d, 5
       add      r8d, r10d
       inc      r9d
       cmp      r9d, 8
       jl       G_M000_IG03
 
G_M000_IG13:                ;; offset=0x017B
       vmovaps  ymm0, ymm3
       vextracti128 xmm1, ymm3
       vpaddd   xmm0, xmm1, xmm0
       vpshufd  xmm1, xmm0, 78
       vpaddd   xmm0, xmm1, xmm0
       vpshufd  xmm1, xmm0, -79
       vpaddd   xmm0, xmm1, xmm0
       vmovd    eax, xmm0
       sub      eax, r8d
 
G_M000_IG14:                ;; offset=0x01A2
       vzeroupper 
       pop      rbx
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
RWD00  	dd	0F0F0F0Fh		; 7.05334e-30
RWD04  	dd	03030303h		; 3.85009e-37
RWD08  	dd	30303030h		; 6.40969e-10
RWD12  	dd	000000E7h ; case G_M000_IG10
       	dd	000000E0h ; case G_M000_IG09
       	dd	000000D9h ; case G_M000_IG08

; Total bytes of code 428

