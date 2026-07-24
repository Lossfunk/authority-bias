import os, subprocess
PUBKEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDSpBh9k28pnlxVvBSmjzpOJypCTk32Whao2LElG1ji4U03ifxpMMx8KISYSWE1qAPvI6GCN++8bOKzh1VBxakLfSUmAKTJzWkW7Ggr4eGoEHQ6OXNLIo8RLsPEYUaE+ynDOhUI/UfO7ICBZ2OZsM3o94DKgmACFM2bg7BZTrBvf3yfk2NFjmtFWJaaDNhTHgbZUC6xc6JsPAmuM5RkRPAk0Whgg/zLYVIiqDr8NAFGdvNi/cMy7g4F9ZiWPQuJZyz5/i1QGWyJKtqqnrDzlrkvdtrXsi1GwrYucNiFMu8B7NajY5hBfyKdQX19aDu8UZ94DA+aoL/QFg7UZ8qHBPE2HjeB5OCBvgWTMRaBHhREj/lXr7rUQGKXtNcNGDGHyJBiyFLKWgzgekKC8iBcZ3xlcXYvS4QzHf7taNU/0geP0WdKmoDD+GQ3M4Gq1X29SFNz9bjzElghpjSX9CG34jTXOqr4ZPmmru+0YSy90wlxpfHK2MI7x3ZyrIwLawktu50ygcupB0bDmGJONmKzaCK+rEzoGMDB3yQkG8uJORPiCaou81DYpRS9EqFa9OhCbEQYPiZ+4XwfYMtka7oBRZyjx56f5DYVXzenHVThKhvw/uGKVtGNVjWwUh0CA7kzwAJrOVqeozyOdzil29xO9WqZ3lvEBkWLqLKdotayFaZfPcJcNynZzCNmakJb8KucRgWzv22iZSzQaw/Hwa/C7XOO50zy3ZurgGQl7b39Euw2vZR9GFFO8Q3z9NlqpUtzQfEADV4fn5asBR5Bnj68mi5Fm9q11pSWmv8hbh7tXtk1GQKgO/djm9ml77vBtTrkTnMoGYyPbVA5Lo2r1ivSTQH7EQWyL94shwPejV+SUdvoRIhIIhnZvbys7SAX3vrfZyRqNR4FmIzWmz/jdGoGqg6H8fm4+BMYDqw0oaQ== majortimberwolf@Abhinavs-MacBook-Pro.local"

print("== whoami / HOME ==")
print(subprocess.run(["bash","-lc","whoami; echo HOME=$HOME; getent passwd root | head -1"], capture_output=True, text=True).stdout)
print("== ensure /root/.ssh ==")
os.makedirs("/root/.ssh", mode=0o700, exist_ok=True)
auth = "/root/.ssh/authorized_keys"
existing = ""
if os.path.exists(auth):
    existing = open(auth).read()
if PUBKEY.split()[1] in existing:
    print("already present in /root/.ssh/authorized_keys")
else:
    with open(auth, "a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(PUBKEY.strip() + "\n")
    os.chmod(auth, 0o600)
    print("appended to /root/.ssh/authorized_keys")
print(subprocess.run(["bash","-lc","ls -la /root/.ssh; head -c 200 /root/.ssh/authorized_keys; echo; wc -l /root/.ssh/authorized_keys"], capture_output=True, text=True).stdout)
print("\n== check sshd is running and ports ==")
print(subprocess.run(["bash","-lc","ps -ef | grep sshd | grep -v grep; ss -tlnp 2>/dev/null | head -10; netstat -tlnp 2>/dev/null | head -10"], capture_output=True, text=True).stdout)
print("\n== sshd config ==")
print(subprocess.run(["bash","-lc","cat /etc/ssh/sshd_config 2>/dev/null | grep -E '^(Port|PermitRootLogin|PubkeyAuthentication|PasswordAuthentication|AuthorizedKeysFile|HostKey)' "], capture_output=True, text=True).stdout)
print("\n== try local ssh loopback ==")
print(subprocess.run(["bash","-lc","ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 -i /root/.ssh/id_ed25519_github_jarvis root@localhost echo SSH_OK 2>&1 | head -10"], capture_output=True, text=True).stdout)
