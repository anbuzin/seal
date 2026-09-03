import vercel.workflow

workflow = vercel.workflow.Workflows(
    sandbox_policy=vercel.workflow.SandboxPolicy(
        passthrough_modules=frozenset(
            {
                "rich",  # annoying terminal detection stuff
                "modelsdotdev",  # sqlite database
            }
        ),
        share_sandboxes=True,
    )
)
