"""Append the local Mac's public key into ~/.ssh/authorized_keys on the remote box."""
import os, subprocess
PUBKEY = """ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDSpBh9k28pnlxVvBSmjzpOJypCTk32Whao2LElG1ji4U03ifxpMMx8KISYSWE1qAPvI6GCN++8bOKzh1VBxakLfSUmAKTJzWkW7Ggr4eGoEHQ6OXNLIo8RLsPEYUaE+ynDOhUI/UfO7ICBZ2OZsM3o94DKgmACFM2bg7BZTrBvf3yfk2NFjmtFWJaaDNhTHgbZUC6xc6JsPAmuM5RkRPAk0Whgg/zLYVIiqDr8NAFGdvNi/cMy7g4F9ZiWPQuJZyz5/i1QGWyJKtqqnrDzlrkvdtrXsi1GwrYucNiFMu8B7NajY5hBfyKdQX19aDu8UZ94DA+aoL/QFg7UZ8qHBPE2HjeB5OCBvgWTMRaBHhREj/lXr7rUQGKXtNcNGDGHyJBiyFLKWgzgekKC8iBcZ3xlcXYvS4QzHf7taNU/0geP0WdKmoDD+GQ3M4Gq1X29SFNz9bjzElghpjSX9CG34jTXOqr4ZPmmru+0YSy90wlxpfHK2MI7x3ZyrIwLawktu50ygcupB0bDmGJONmKzaCK+rEzoGMDB3yQkG8uJORPiCaou81DYpRS9EqFa9OhCbEQYPiZ+4XwfYMtka7oBRZyjx56f5DYVXzenHVThKhvw/uGKVtGNVjWwUh0CA7kzwAJrOVqeozyOdzil29xO9WqZ3lvEBkWLqLKdotayFaZfPcJcNynZzCNmakJb8KucRgWzv22iZSzQaw/Hwa/C7XOO50zy3ZurgGQl7b39Euw2vZR9GFFO8Q3z9NlqpUtzQfEADV4fn5asBR5Bnj68mi5Fm9q11pSWmv8hbh7tXtk1GQKgO/djm9ml77vBtTrkTnMoGYyPbVA5Lo2r1ivSTQH7EQWyL94shwPejV+SUdvoRIhIIhnZvbys7SAX3vrfZyRqNR4FmIzWmz/jdGoGqg6H8fm4+BMYDqw0oaQ== majortimberwolf@Abhinavs-MacBook-Pro.local"""

ssh_dir = os.path.expanduser("~/.ssh")
os.makedirs(ssh_dir, mode=0o700, exist_ok=True)
auth = os.path.join(ssh_dir, "authorized_keys")
existing = ""
if os.path.exists(auth):
    existing = open(auth).read()
if PUBKEY.split()[1] in existing:
    print("already present")
else:
    with open(auth, "a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(PUBKEY.strip() + "\n")
    os.chmod(auth, 0o600)
    print("appended")

print("== sshd_config root login / pubkey settings ==")
print(subprocess.run(["bash","-lc","grep -E '^(PermitRootLogin|PubkeyAuthentication|PasswordAuthentication|AuthorizedKeysFile)' /etc/ssh/sshd_config 2>/dev/null"], capture_output=True, text=True).stdout)
print("== which sshd is up ==")
print(subprocess.run(["bash","-lc","ss -tlnp 2>/dev/null | grep -E ':22 |:2222 ' || netstat -tlnp 2>/dev/null | grep -E ':22 |:2222 '"], capture_output=True, text=True).stdout)
print("== ip / port info ==")
print(subprocess.run(["bash","-lc","ip addr show 2>/dev/null | head -40"], capture_output=True, text=True).stdout)
print("== authorized_keys after ==")
print(subprocess.run(["bash","-lc","wc -l ~/.ssh/authorized_keys; ls -la ~/.ssh"], capture_output=True, text=True).stdout)
