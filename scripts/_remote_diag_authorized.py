import os, subprocess
print("auth_keys content (first 4000):")
print(subprocess.run(["bash","-lc","cat /root/.ssh/authorized_keys"], capture_output=True, text=True).stdout[:4000])
print("\nperms:")
print(subprocess.run(["bash","-lc","ls -la /root /root/.ssh /root/.ssh/authorized_keys"], capture_output=True, text=True).stdout)
print("\nsshd full config:")
print(subprocess.run(["bash","-lc","sshd -T 2>/dev/null | grep -E '^(authorizedkeysfile|permitrootlogin|pubkeyauthentication|passwordauthentication|hostkey|port|listenaddress)'"], capture_output=True, text=True).stdout)
print("\ntailing sshd journal:")
print(subprocess.run(["bash","-lc","journalctl -u ssh -n 20 2>/dev/null; tail -n 30 /var/log/auth.log 2>/dev/null"], capture_output=True, text=True).stdout)
